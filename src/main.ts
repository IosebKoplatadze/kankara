import { DoublePendulum } from './physics/DoublePendulum';
import { Renderer } from './visualization/Renderer';
import { NEATEvolution } from './neat/NEATEvolution';
import { WorkerPool } from './workers/WorkerPool';

class KankaraApp {
    private canvas: HTMLCanvasElement;
    private renderer: Renderer;
    private evolution: NEATEvolution;
    private workerPool: WorkerPool;
    private animationId: number = 0;
    private isRunning: boolean = false;
    private lastTime: number = 0;
    private fps: number = 0;
    private frameCount: number = 0;
    private fpsUpdateTime: number = 0;

    constructor() {
        this.canvas = document.getElementById('simulation-canvas') as HTMLCanvasElement;
        this.renderer = new Renderer(this.canvas);
        this.evolution = new NEATEvolution(50); // Default population size
        this.workerPool = new WorkerPool(navigator.hardwareConcurrency || 4);
        
        this.setupEventListeners();
        this.updateUI();
    }

    private setupEventListeners(): void {
        const startBtn = document.getElementById('start-evolution') as HTMLButtonElement;
        const pauseBtn = document.getElementById('pause-evolution') as HTMLButtonElement;
        const resetBtn = document.getElementById('reset-simulation') as HTMLButtonElement;
        const speedSlider = document.getElementById('sim-speed') as HTMLInputElement;
        const populationInput = document.getElementById('population-size') as HTMLInputElement;

        startBtn.addEventListener('click', () => this.startEvolution());
        pauseBtn.addEventListener('click', () => this.pauseEvolution());
        resetBtn.addEventListener('click', () => this.resetSimulation());
        
        speedSlider.addEventListener('input', (e) => {
            const target = e.target as HTMLInputElement;
            const speedValue = document.getElementById('speed-value');
            if (speedValue) speedValue.textContent = `${parseFloat(target.value).toFixed(1)}x`;
        });

        populationInput.addEventListener('change', (e) => {
            const target = e.target as HTMLInputElement;
            const newSize = parseInt(target.value);
            if (newSize >= 10 && newSize <= 500) {
                this.evolution.setPopulationSize(newSize);
            }
        });
    }

    private async startEvolution(): Promise<void> {
        if (!this.isRunning) {
            this.isRunning = true;
            await this.workerPool.initialize();
            this.animate();
        }
    }

    private pauseEvolution(): void {
        this.isRunning = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
    }

    private resetSimulation(): void {
        this.pauseEvolution();
        this.evolution.reset();
        this.renderer.clear();
        this.updateUI();
    }

    private animate = (currentTime: number = 0): void => {
        if (!this.isRunning) return;

        const deltaTime = currentTime - this.lastTime;
        this.lastTime = currentTime;

        // Update FPS counter
        this.frameCount++;
        if (currentTime - this.fpsUpdateTime >= 1000) {
            this.fps = Math.round((this.frameCount * 1000) / (currentTime - this.fpsUpdateTime));
            this.frameCount = 0;
            this.fpsUpdateTime = currentTime;
            this.updateFPS();
        }

        // Update evolution
        this.evolution.update(deltaTime);

        // Render current state
        this.render();

        // Update UI
        this.updateUI();

        this.animationId = requestAnimationFrame(this.animate);
    };

    private render(): void {
        this.renderer.clear();
        
        const viewMode = (document.getElementById('view-mode') as HTMLSelectElement).value;
        
        switch (viewMode) {
            case 'best':
                this.renderBestIndividual();
                break;
            case 'population':
                this.renderPopulation();
                break;
            case 'comparison':
                this.renderComparison();
                break;
        }
    }

    private renderBestIndividual(): void {
        const best = this.evolution.getBestIndividual();
        if (best && best.pendulum) {
            this.renderer.drawPendulum(best.pendulum, { alpha: 1.0, color: '#00ff00' });
        }
    }

    private renderPopulation(): void {
        const population = this.evolution.getPopulation();
        population.forEach((individual, index) => {
            if (individual.pendulum) {
                const alpha = 0.1 + (individual.fitness / this.evolution.getMaxFitness()) * 0.9;
                this.renderer.drawPendulum(individual.pendulum, { 
                    alpha, 
                    color: `hsl(${(index / population.length) * 360}, 70%, 50%)` 
                });
            }
        });
    }

    private renderComparison(): void {
        // Render best few individuals side by side
        const best = this.evolution.getBestIndividuals(4);
        best.forEach((individual, index) => {
            if (individual.pendulum) {
                const offset = (index - best.length / 2) * 200;
                this.renderer.drawPendulum(individual.pendulum, { 
                    alpha: 1.0, 
                    color: `hsl(${index * 90}, 80%, 60%)`,
                    offsetX: offset
                });
            }
        });
    }

    private updateUI(): void {
        const generationSpan = document.getElementById('generation');
        const bestFitnessSpan = document.getElementById('best-fitness');
        const avgFitnessSpan = document.getElementById('avg-fitness');
        const speciesSpan = document.getElementById('species-count');
        const workersSpan = document.getElementById('workers');

        if (generationSpan) generationSpan.textContent = this.evolution.getGeneration().toString();
        if (bestFitnessSpan) bestFitnessSpan.textContent = this.evolution.getBestFitness().toFixed(2);
        if (avgFitnessSpan) avgFitnessSpan.textContent = this.evolution.getAverageFitness().toFixed(2);
        if (speciesSpan) speciesSpan.textContent = this.evolution.getSpeciesCount().toString();
        if (workersSpan) workersSpan.textContent = this.workerPool.getActiveWorkers().toString();
    }

    private updateFPS(): void {
        const fpsSpan = document.getElementById('fps');
        if (fpsSpan) fpsSpan.textContent = this.fps.toString();
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new KankaraApp();
});