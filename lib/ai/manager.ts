/**
 * AI Model Manager - Central orchestration for multi-provider AI
 * 
 * Features:
 * - Multi-provider support (OpenRouter, Gemini)
 * - Use case-based model selection
 * - Automatic fallback handling
 * - Health monitoring (delegated)
 * - Unified interface for all AI operations
 */

import type {
  ProviderName,
  ModelInfo,
  CompletionOptions,
  CompletionResult,
  StreamChunk,
  HealthCheckResult,
  ProviderStatus,
  AIManagerConfig,
  UseCaseConfig,
  UseCase,
} from './types'

import { AIManagerConfigSchema } from './types'
import { logger } from './utils/logger'
import { DEFAULT_USE_CASE_MODELS } from './model-configs'
import { HealthMonitor } from './health-monitor'
import { AIProviderFactory, AIProvider } from './provider-factory'

let logQueryFn: any = null
try {
  const loggerModule = require('./query-logger')
  logQueryFn = loggerModule.logQuery
} catch (e) {
  logQueryFn = async () => { }
}

export class AIModelManager {
  private providers: Map<ProviderName, AIProvider> = new Map()
  private useCaseConfigs: Map<UseCase, UseCaseConfig> = new Map()
  private healthMonitor: HealthMonitor
  private initialized: boolean = false

  constructor(config?: AIManagerConfig) {
    this.healthMonitor = new HealthMonitor()
    if (config) {
      this.initialize(config)
    }
  }

  /**
   * Initialize the manager with configuration
   */
  initialize(config: AIManagerConfig): void {
    // Validate configuration
    const validatedConfig = AIManagerConfigSchema.parse(config)

    // Configure logging
    logger.configure({
      level: validatedConfig.logLevel,
      enabled: validatedConfig.enableLogging,
    })

    // Initialize providers via Factory
    for (const providerConfig of validatedConfig.providers) {
      const provider = AIProviderFactory.createProvider(providerConfig)
      if (provider) {
        this.providers.set(provider.providerName, provider)
        this.healthMonitor.registerProvider(provider)

        logger.info('AIManager', `Initialized provider: ${provider.providerName}`, {
          models: provider.getModels().length,
        })
      }
    }

    // Configure use cases
    if (validatedConfig.useCases) {
      for (const useCaseConfig of validatedConfig.useCases) {
        this.useCaseConfigs.set(useCaseConfig.useCase, useCaseConfig)
      }
    }

    // Start health checks if enabled
    if (validatedConfig.enableHealthChecks) {
      this.healthMonitor.start(validatedConfig.healthCheckIntervalMs)
    }

    this.initialized = true
    logger.info('AIManager', 'Initialization complete', {
      providers: this.providers.size,
      useCases: this.useCaseConfigs.size,
    })
  }

  /**
   * Initialize from environment variables
   */
  initializeFromEnv(): void {
    const providers = AIProviderFactory.createFromEnv()

    if (providers.length === 0) {
      throw new Error('No AI providers configured. Set GOOGLE_GENERATIVE_AI_API_KEY or GEMINI_API_KEY.')
    }

    // Initialize with discovered providers
    // We construct a config object to reuse the main initialize logic or just set directly
    // Setting directly here for simplicity as factory returns instances
    providers.forEach(p => {
      this.providers.set(p.providerName, p)
      this.healthMonitor.registerProvider(p)
      logger.info('AIManager', `Initialized provider from env: ${p.providerName}`)
    })

    // Default configuration for env initialization
    this.healthMonitor.start(parseInt(process.env.AI_HEALTH_CHECK_INTERVAL || '60000', 10))

    logger.configure({
      level: (process.env.AI_LOG_LEVEL as 'debug' | 'info' | 'warn' | 'error') || 'info',
      enabled: process.env.AI_LOGGING !== 'false',
    })

    this.initialized = true
  }

  /**
   * Get all available models across all providers
   */
  getAllModels(): ModelInfo[] {
    const models: ModelInfo[] = []
    for (const provider of this.providers.values()) {
      models.push(...provider.getModels())
    }
    return models
  }

  /**
   * Get models from a specific provider
   */
  getProviderModels(providerName: ProviderName): ModelInfo[] {
    const provider = this.providers.get(providerName)
    if (provider) {
      return provider.getModels()
    }
    return []
  }

  /**
   * Get a specific model by full ID (provider:model)
   */
  getModel(fullModelId: string): ModelInfo | undefined {
    const [providerName, modelId] = this.parseModelId(fullModelId)
    const provider = this.providers.get(providerName)
    if (provider) {
      return provider.getModel(modelId)
    }
    return undefined
  }

  /**
   * Parse a full model ID into provider and model parts
   */
  private parseModelId(fullModelId: string): [ProviderName, string] {
    const colonIndex = fullModelId.indexOf(':')
    if (colonIndex === -1) {
      // Auto-detect provider from model name prefix
      if (fullModelId.startsWith('gemini-')) {
        return ['gemini', fullModelId]
      }
      // Default to gemini for models without explicit provider
      return ['gemini', fullModelId]
    }
    return [
      fullModelId.substring(0, colonIndex) as ProviderName,
      fullModelId.substring(colonIndex + 1),
    ]
  }

  /**
   * Complete a chat with automatic fallback
   */
  async complete(options: CompletionOptions & { useCase?: UseCase }): Promise<CompletionResult> {
    const startTime = Date.now()
    const useCase = options.useCase || 'chat'
    const firstPrompt = Array.isArray(options.messages) && options.messages.length > 0
      ? (typeof options.messages[0].content === 'string' ? options.messages[0].content : JSON.stringify(options.messages[0].content))
      : ''
    const modelsToTry = this.getModelsForRequest(options)

    let lastError: Error | undefined

    for (const fullModelId of modelsToTry) {
      const [providerName, modelId] = this.parseModelId(fullModelId)
      const provider = this.providers.get(providerName)

      if (!provider) {
        logger.warn('AIManager', `Provider not found: ${providerName}`)
        continue
      }

      const status = this.healthMonitor.getProviderStatus(providerName)
      if (status && status.consecutiveFailures >= 5) {
        logger.warn('AIManager', `Skipping unhealthy provider: ${providerName}`)
        continue
      }

      try {
        const result = await provider.complete({
          ...options,
          model: modelId,
        })

        if (logQueryFn) {
          logQueryFn({
            useCase,
            provider: providerName,
            model: modelId,
            prompt: firstPrompt.slice(0, 200),
            success: true,
            latencyMs: Date.now() - startTime,
            tokensUsed: result.usage?.totalTokens
          }).catch(() => { })
        }

        return result
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error))

        // Let HealthMonitor know about failure implicitly through health checks, 
        // or we could expose a reportFailure method on it. 
        // For now, we rely on the background health checks to catch persistent issues,
        // but we can log the fallback here.

        const nextModel = modelsToTry[modelsToTry.indexOf(fullModelId) + 1]
        if (nextModel) {
          logger.fallback(fullModelId, nextModel, lastError.message)
        }
      }
    }

    if (logQueryFn) {
      logQueryFn({
        useCase,
        provider: 'unknown',
        model: modelsToTry[0] || 'unknown',
        prompt: firstPrompt.slice(0, 200),
        success: false,
        error: lastError?.message || 'All models failed',
        latencyMs: Date.now() - startTime
      }).catch(() => { })
    }

    throw lastError || new Error('All models failed')
  }

  /**
   * Stream a chat completion with automatic fallback
   */
  async *stream(
    options: CompletionOptions & { useCase?: UseCase }
  ): AsyncGenerator<StreamChunk> {
    const modelsToTry = this.getModelsForRequest(options)

    let lastError: Error | undefined

    for (const fullModelId of modelsToTry) {
      const [providerName, modelId] = this.parseModelId(fullModelId)
      const provider = this.providers.get(providerName)

      if (!provider) continue

      const status = this.healthMonitor.getProviderStatus(providerName)
      if (status && status.consecutiveFailures >= 5) continue

      try {
        for await (const chunk of provider.stream({
          ...options,
          model: modelId,
        })) {
          yield chunk
        }
        return
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error))
        const nextModel = modelsToTry[modelsToTry.indexOf(fullModelId) + 1]
        if (nextModel) {
          logger.fallback(fullModelId, nextModel, lastError.message)
        }
      }
    }

    throw lastError || new Error('All models failed')
  }

  /**
   * Get ordered list of models to try for a request
   * Uses DEFAULT_USE_CASE_MODELS with optimal Gemini models
   */
  private getModelsForRequest(options: CompletionOptions & { useCase?: UseCase }): string[] {
    // If specific model requested, use it with fallbacks
    if (options.model) {
      // If it's a full model ID, use it directly
      if (options.model.includes(':')) {
        return [options.model]
      }

      // Check all providers for this model
      for (const [name, provider] of this.providers.entries()) {
        if (provider.hasModel(options.model)) {
          return [`${name}:${options.model}`]
        }
      }

      // Default fall-through if not found (likely will fail later but keeps logic simple)
      return [`gemini:${options.model}`]
    }

    // Get models based on use case
    const useCase = options.useCase || 'chat'
    const useCaseConfig = DEFAULT_USE_CASE_MODELS[useCase] || DEFAULT_USE_CASE_MODELS.chat

    return [useCaseConfig.primary, ...useCaseConfig.fallbacks]
  }

  /**
   * Configure a use case with specific models
   */
  configureUseCase(config: UseCaseConfig): void {
    this.useCaseConfigs.set(config.useCase, config)
    logger.info('AIManager', `Configured use case: ${config.useCase}`, {
      primary: config.primaryModel,
      fallbacks: config.fallbackModels.length,
    })
  }

  /**
   * Run health checks on all providers
   */
  async runHealthChecks(): Promise<HealthCheckResult[]> {
    return this.healthMonitor.runHealthChecks()
  }

  /**
   * Get provider statuses
   */
  getProviderStatuses(): ProviderStatus[] {
    return this.healthMonitor.getAllStatuses()
  }

  /**
   * Get a specific provider status
   */
  getProviderStatus(providerName: ProviderName): ProviderStatus | undefined {
    return this.healthMonitor.getProviderStatus(providerName)
  }

  /**
   * Check if a provider is healthy
   */
  isProviderHealthy(providerName: ProviderName): boolean {
    const status = this.healthMonitor.getProviderStatus(providerName)
    return status?.healthy ?? false
  }

  /**
   * Get healthy providers
   */
  getHealthyProviders(): ProviderName[] {
    return this.healthMonitor.getAllStatuses()
      .filter(status => status.healthy)
      .map(status => status.name)
  }

  /**
   * Stop health checks and cleanup
   */
  shutdown(): void {
    this.healthMonitor.stop()
    logger.info('AIManager', 'Shutdown complete')
  }

  /**
   * Check if manager is initialized
   */
  isInitialized(): boolean {
    return this.initialized
  }
}

// Singleton instance
let managerInstance: AIModelManager | null = null

/**
 * Get or create the singleton AI Manager instance
 */
export function getAIManager(): AIModelManager {
  if (!managerInstance) {
    managerInstance = new AIModelManager()
    managerInstance.initializeFromEnv()
  }
  return managerInstance
}

/**
 * Create a new AI Manager instance with custom config
 */
export function createAIManager(config: AIManagerConfig): AIModelManager {
  return new AIModelManager(config)
}

// Re-export providers for direct use if needed
import { GeminiProvider } from './providers/gemini'
export { GeminiProvider }
