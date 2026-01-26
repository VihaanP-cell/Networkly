import { logger } from './utils/logger'
import type { HealthCheckResult, ProviderName, ProviderStatus } from './types'

export interface HealthCheckableProvider {
    providerName: ProviderName
    healthCheck(model?: string): Promise<HealthCheckResult>
    getModels(): Array<{ id: string }>
}

export class HealthMonitor {
    private healthCheckInterval?: ReturnType<typeof setInterval>
    private providerStatuses: Map<ProviderName, ProviderStatus> = new Map()
    private providers: Map<ProviderName, HealthCheckableProvider> = new Map()

    constructor(
        private intervalMs: number = 60000,
        initialProviders: HealthCheckableProvider[] = []
    ) {
        initialProviders.forEach(p => this.registerProvider(p))
    }

    registerProvider(provider: HealthCheckableProvider) {
        this.providers.set(provider.providerName, provider)

        // Initialize status if not exists
        if (!this.providerStatuses.has(provider.providerName)) {
            this.providerStatuses.set(provider.providerName, {
                name: provider.providerName,
                healthy: true, // Assume healthy initially
                lastCheck: new Date(),
                consecutiveFailures: 0,
                averageLatencyMs: 0,
                modelsHealthy: provider.getModels().length,
                modelsUnhealthy: 0,
            })
        }
    }

    start(intervalMs?: number) {
        if (intervalMs) this.intervalMs = intervalMs

        // Stop existing if any
        this.stop()

        this.healthCheckInterval = setInterval(async () => {
            await this.runHealthChecks()
        }, this.intervalMs)

        logger.info('HealthMonitor', 'Started health checks', { intervalMs: this.intervalMs })
    }

    stop() {
        if (this.healthCheckInterval) {
            clearInterval(this.healthCheckInterval)
            this.healthCheckInterval = undefined
        }
    }

    async runHealthChecks(): Promise<HealthCheckResult[]> {
        const results: HealthCheckResult[] = []

        for (const [name, provider] of this.providers.entries()) {
            try {
                const result = await provider.healthCheck()
                results.push(result)

                this.updateProviderStatus(name, result)

            } catch (error) {
                logger.error('HealthMonitor', `Health check failed for ${name}`, { error: String(error) })

                // Create a failure result
                const failureResult: HealthCheckResult = {
                    provider: name,
                    model: 'unknown',
                    healthy: false,
                    latencyMs: 0,
                    error: error instanceof Error ? error.message : String(error),
                    timestamp: new Date()
                }
                this.updateProviderStatus(name, failureResult)
                results.push(failureResult)
            }
        }

        return results
    }

    private updateProviderStatus(name: ProviderName, result: HealthCheckResult) {
        const status = this.providerStatuses.get(name)
        if (!status) return

        status.lastCheck = new Date()

        if (result.healthy) {
            status.healthy = true
            status.consecutiveFailures = 0
            // Simple moving average for latency
            status.averageLatencyMs = (status.averageLatencyMs * 0.7) + (result.latencyMs * 0.3)
        } else {
            status.consecutiveFailures++
            if (status.consecutiveFailures >= 3) {
                status.healthy = false
            }
        }

        this.providerStatuses.set(name, status)
    }

    getProviderStatus(name: ProviderName): ProviderStatus | undefined {
        return this.providerStatuses.get(name)
    }

    getAllStatuses(): ProviderStatus[] {
        return Array.from(this.providerStatuses.values())
    }
}
