import { DoublePendulum } from './physics/DoublePendulum';
import { Renderer } from './visualization/Renderer';

class SimpleDemo {
    private canvas: HTMLCanvasElement;
    private renderer: Renderer;
    private pendulum: DoublePendulum;
    private animationId: number = 0;
    private isRunning: boolean = false;
    private lastTime: number = 0;
    private fps: number = 0;
    private frameCount: number = 0;
    private fpsUpdateTime: number = 0;

    constructor() {
        this.canvas = document.getElementById('simulation-canvas') as HTMLCanvasElement;
        this.renderer = new Renderer(this.canvas);
        
        // Create pendulum with interesting initial conditions
        this.pendulum = new DoublePendulum({
            theta1: Math.PI * 0.7,
            theta2: Math.PI * 0.9,
            omega1: 0.5,
            omega2: -0.3
        });
        
        this.setupEventListeners();
        this.startRenderLoop();
    }

    private setupEventListeners(): void {
        const startBtn = document.getElementById('start-evolution') as HTMLButtonElement;
        const pauseBtn = document.getElementById('pause-evolution') as HTMLButtonElement;
        const resetBtn = document.getElementById('reset-simulation') as HTMLButtonElement;

        startBtn.addEventListener('click', () => this.startSimulation());
        pauseBtn.addEventListener('click', () => this.pauseSimulation());
        resetBtn.addEventListener('click', () => this.resetSimulation());
    }

    private startSimulation(): void {
        this.isRunning = true;
    }

    private pauseSimulation(): void {
        this.isRunning = false;
    }

    private resetSimulation(): void {
        this.pendulum.reset({
            theta1: Math.PI * 0.7 + (Math.random() - 0.5) * 0.4,
            theta2: Math.PI * 0.9 + (Math.random() - 0.5) * 0.4,
            omega1: (Math.random() - 0.5) * 1.0,
            omega2: (Math.random() - 0.5) * 1.0
        });
    }

    private startRenderLoop(): void {
        const animate = (currentTime: number = 0) => {
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

            // Update physics if running
            if (this.isRunning && deltaTime > 0) {
                const dt = Math.min(deltaTime / 1000, 1/30); // Cap at 30 FPS for stability
                this.pendulum.update(dt);
            }

            // Always render
            this.render();

            // Update UI
            this.updateUI();

            requestAnimationFrame(animate);
        };
        
        requestAnimationFrame(animate);
    }

    private render(): void {
        this.renderer.clear();
        this.renderer.drawPendulum(this.pendulum, { 
            alpha: 1.0, 
            color: this.isRunning ? '#00ff00' : '#666666'
        });
    }

    private updateUI(): void {
        const bestFitnessSpan = document.getElementById('best-fitness');
        const avgFitnessSpan = document.getElementById('avg-fitness');

        const fitness = this.pendulum.calculateFitness();
        const energy = this.pendulum.getEnergy();
        
        if (bestFitnessSpan) bestFitnessSpan.textContent = fitness.toFixed(2);
        if (avgFitnessSpan) avgFitnessSpan.textContent = energy.toFixed(2);
    }

    private updateFPS(): void {
        const fpsSpan = document.getElementById('fps');
        if (fpsSpan) fpsSpan.textContent = this.fps.toString();
    }
}

// Initialize the simple demo
document.addEventListener('DOMContentLoaded', () => {
    new SimpleDemo();
});