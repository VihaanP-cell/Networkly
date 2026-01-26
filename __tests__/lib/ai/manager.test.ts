
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { AIModelManager } from '@/lib/ai/manager'
import { HealthMonitor } from '@/lib/ai/health-monitor'
import { AIProviderFactory } from '@/lib/ai/provider-factory'

// Mock dependencies
vi.mock('@/lib/ai/provider-factory')
vi.mock('@/lib/ai/utils/logger', () => ({
    logger: {
        configure: vi.fn(),
        info: vi.fn(),
        warn: vi.fn(),
        error: vi.fn(),
        fallback: vi.fn(),
        request: vi.fn(),
        response: vi.fn(),
    }
}))

// Mock HealthMonitor with a factory
const mockHealthMonitorInstance = {
    registerProvider: vi.fn(),
    start: vi.fn(),
    stop: vi.fn(),
    runHealthChecks: vi.fn(),
    getProviderStatus: vi.fn().mockReturnValue({ healthy: true, consecutiveFailures: 0 }),
    getAllStatuses: vi.fn().mockReturnValue([{ name: 'gemini', healthy: true }]),
}

vi.mock('@/lib/ai/health-monitor', () => ({
    HealthMonitor: class {
        constructor() {
            return mockHealthMonitorInstance
        }
    }
}))

describe('AIModelManager', () => {
    let manager: AIModelManager
    let mockProvider: any

    beforeEach(() => {
        vi.clearAllMocks()

        // Setup mock provider
        mockProvider = {
            providerName: 'gemini',
            complete: vi.fn().mockResolvedValue({
                id: 'test-id',
                content: 'test response',
                usage: { totalTokens: 10 }
            }),
            stream: vi.fn(),
            getModels: vi.fn().mockReturnValue([{ id: 'gemini-pro' }]),
            getModel: vi.fn(),
            hasModel: vi.fn().mockReturnValue(true),
            healthCheck: vi.fn().mockResolvedValue({ healthy: true }),
        }

        // Mock Factory to return our mock provider
        vi.spyOn(AIProviderFactory, 'createProvider').mockReturnValue(mockProvider)
        vi.spyOn(AIProviderFactory, 'createFromEnv').mockReturnValue([mockProvider])

        manager = new AIModelManager()
    })

    afterEach(() => {
        vi.restoreAllMocks()
    })

    describe('initialization', () => {
        it('should initialize from environment', () => {
            manager.initializeFromEnv()
            expect(AIProviderFactory.createFromEnv).toHaveBeenCalled()
            expect(manager.isInitialized()).toBe(true)
        })

        it('should throw if no providers found in env', () => {
            vi.spyOn(AIProviderFactory, 'createFromEnv').mockReturnValue([])
            expect(() => manager.initializeFromEnv()).toThrow('No AI providers configured')
        })
    })

    describe('complete', () => {
        beforeEach(() => {
            manager.initializeFromEnv()
        })

        it('should route to the correct provider', async () => {
            await manager.complete({
                model: 'gemini:gemini-pro',
                messages: [{ role: 'user', content: 'hello' }]
            })

            expect(mockProvider.complete).toHaveBeenCalledWith(expect.objectContaining({
                model: 'gemini-pro'
            }))
        })

        it('should use basic fallback when provider fails', async () => {
            // First call fails
            mockProvider.complete.mockRejectedValueOnce(new Error('API Error'))

            await expect(manager.complete({
                model: 'gemini:gemini-pro',
                messages: [{ role: 'user', content: 'hello' }]
            })).rejects.toThrow('API Error')
        })
    })

    describe('health checks', () => {
        beforeEach(() => {
            manager.initializeFromEnv()
        })

        it('should delegate health checks to HealthMonitor', async () => {
            await manager.runHealthChecks()
            // We can check if our mock instance method was called
            expect(mockHealthMonitorInstance.runHealthChecks).toHaveBeenCalled()
        })
    })
})
