import { Individual } from '../neat/NEATEvolution';

export interface WorkerTask {
    id: string;
    individual: Individual;
    simulationTime: number;
}

export interface WorkerResult {
    taskId: string;
    fitness: number;
    simulationData?: any;
}

export class WorkerPool {
    private workers: Worker[] = [];
    private availableWorkers: Worker[] = [];
    private busyWorkers: Set<Worker> = new Set();
    private taskQueue: WorkerTask[] = [];
    private pendingTasks: Map<string, (result: WorkerResult) => void> = new Map();
    private workerCount: number;

    constructor(workerCount: number) {
        this.workerCount = Math.min(workerCount, navigator.hardwareConcurrency || 4);
    }

    public async initialize(): Promise<void> {
        if (this.workers.length > 0) return; // Already initialized

        for (let i = 0; i < this.workerCount; i++) {
            const worker = new Worker(
                new URL('./SimulationWorker.ts', import.meta.url),
                { type: 'module' }
            );
            
            worker.onmessage = (event) => this.handleWorkerMessage(worker, event);
            worker.onerror = (error) => this.handleWorkerError(worker, error);
            
            this.workers.push(worker);
            this.availableWorkers.push(worker);
        }

        console.log(`Initialized ${this.workerCount} workers for parallel simulation`);
    }

    public async evaluateIndividuals(
        individuals: Individual[],
        simulationTime: number = 5000
    ): Promise<WorkerResult[]> {
        const results: WorkerResult[] = [];
        const promises: Promise<WorkerResult>[] = [];

        for (const individual of individuals) {
            const task: WorkerTask = {
                id: `task_${individual.id}_${Date.now()}`,
                individual,
                simulationTime
            };

            const promise = new Promise<WorkerResult>((resolve) => {
                this.pendingTasks.set(task.id, resolve);
                this.queueTask(task);
            });

            promises.push(promise);
        }

        const allResults = await Promise.all(promises);
        return allResults;
    }

    private queueTask(task: WorkerTask): void {
        const worker = this.availableWorkers.pop();
        
        if (worker) {
            this.assignTaskToWorker(worker, task);
        } else {
            this.taskQueue.push(task);
        }
    }

    private assignTaskToWorker(worker: Worker, task: WorkerTask): void {
        this.busyWorkers.add(worker);
        
        // Send task to worker
        worker.postMessage({
            type: 'evaluate',
            task
        });
    }

    private handleWorkerMessage(worker: Worker, event: MessageEvent): void {
        const { type, result, error } = event.data;

        if (type === 'result') {
            this.handleTaskResult(worker, result);
        } else if (type === 'error') {
            this.handleTaskError(worker, error);
        }
    }

    private handleTaskResult(worker: Worker, result: WorkerResult): void {
        const resolver = this.pendingTasks.get(result.taskId);
        if (resolver) {
            resolver(result);
            this.pendingTasks.delete(result.taskId);
        }

        this.releaseWorker(worker);
    }

    private handleTaskError(worker: Worker, error: any): void {
        console.error('Worker task error:', error);
        this.releaseWorker(worker);
    }

    private handleWorkerError(worker: Worker, error: ErrorEvent): void {
        console.error('Worker error:', error);
        this.releaseWorker(worker);
    }

    private releaseWorker(worker: Worker): void {
        this.busyWorkers.delete(worker);
        
        // Process next task in queue
        const nextTask = this.taskQueue.shift();
        if (nextTask) {
            this.assignTaskToWorker(worker, nextTask);
        } else {
            this.availableWorkers.push(worker);
        }
    }

    public getActiveWorkers(): number {
        return this.busyWorkers.size;
    }

    public getTotalWorkers(): number {
        return this.workerCount;
    }

    public getQueueSize(): number {
        return this.taskQueue.length;
    }

    public terminate(): void {
        for (const worker of this.workers) {
            worker.terminate();
        }
        
        this.workers = [];
        this.availableWorkers = [];
        this.busyWorkers.clear();
        this.taskQueue = [];
        this.pendingTasks.clear();
    }

    /**
     * Batch evaluate multiple individuals efficiently
     */
    public async batchEvaluate(
        individuals: Individual[],
        simulationTime: number = 5000,
        batchSize?: number
    ): Promise<WorkerResult[]> {
        const actualBatchSize = batchSize || this.workerCount;
        const results: WorkerResult[] = [];

        // Process in batches to avoid overwhelming the workers
        for (let i = 0; i < individuals.length; i += actualBatchSize) {
            const batch = individuals.slice(i, i + actualBatchSize);
            const batchResults = await this.evaluateIndividuals(batch, simulationTime);
            results.push(...batchResults);
        }

        return results;
    }

    /**
     * Dynamically adjust worker count based on system performance
     */
    public adjustWorkerCount(newCount: number): void {
        const targetCount = Math.min(newCount, navigator.hardwareConcurrency || 4);
        
        if (targetCount > this.workerCount) {
            // Add workers
            const workersToAdd = targetCount - this.workerCount;
            for (let i = 0; i < workersToAdd; i++) {
                const worker = new Worker(
                    new URL('./SimulationWorker.ts', import.meta.url),
                    { type: 'module' }
                );
                
                worker.onmessage = (event) => this.handleWorkerMessage(worker, event);
                worker.onerror = (error) => this.handleWorkerError(worker, error);
                
                this.workers.push(worker);
                this.availableWorkers.push(worker);
            }
        } else if (targetCount < this.workerCount) {
            // Remove workers
            const workersToRemove = this.workerCount - targetCount;
            for (let i = 0; i < workersToRemove && this.availableWorkers.length > 0; i++) {
                const worker = this.availableWorkers.pop();
                if (worker) {
                    worker.terminate();
                    const index = this.workers.indexOf(worker);
                    if (index !== -1) {
                        this.workers.splice(index, 1);
                    }
                }
            }
        }
        
        this.workerCount = targetCount;
    }
}