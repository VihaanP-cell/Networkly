import { GeminiProvider, createGeminiProvider } from './providers/gemini'
import { logger } from './utils/logger'
import type { ProviderConfig, ProviderName } from './types'

// Define a common interface for providers
// This should match the structure of GeminiProvider and others
export interface AIProvider {
    providerName: ProviderName
    complete(options: any): Promise<any>
    stream(options: any): AsyncGenerator<any>
    getModels(): any[]
    getModel(modelId: string): any
    hasModel(modelId: string): boolean
    healthCheck(model?: string): Promise<any>
}

export class AIProviderFactory {
    static createProvider(config: ProviderConfig): AIProvider | null {
        if (!config.enabled) return null

        try {
            switch (config.name) {
                case 'gemini':
                    return new GeminiProvider(config)
                default:
                    logger.warn('AIProviderFactory', `Unknown provider: ${config.name}`)
                    return null
            }
        } catch (error) {
            logger.error('AIProviderFactory', `Failed to initialize provider: ${config.name}`, {
                error: String(error)
            })
            return null
        }
    }

    static createFromEnv(): AIProvider[] {
        const providers: AIProvider[] = []

        // Gemini
        const geminiKey = process.env.GOOGLE_GENERATIVE_AI_API_KEY || process.env.GEMINI_API_KEY
        if (geminiKey) {
            providers.push(createGeminiProvider(geminiKey, {
                defaultModel: process.env.GEMINI_DEFAULT_MODEL || 'gemini-2.0-flash',
                timeout: parseInt(process.env.AI_TIMEOUT || '60000', 10),
                maxRetries: parseInt(process.env.AI_MAX_RETRIES || '3', 10),
            }))
        }

        return providers
    }
}
