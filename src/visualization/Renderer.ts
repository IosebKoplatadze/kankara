import { DoublePendulum } from '../physics/DoublePendulum';

export interface RenderOptions {
    alpha?: number;
    color?: string;
    offsetX?: number;
    offsetY?: number;
    scale?: number;
}

export class Renderer {
    private canvas: HTMLCanvasElement;
    private ctx: CanvasRenderingContext2D;
    private centerX: number;
    private centerY: number;
    private scale: number = 100; // pixels per meter

    constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d')!;
        this.centerX = canvas.width / 2;
        this.centerY = canvas.height / 3; // Position pivot point in upper third
        
        this.setupCanvas();
    }

    private setupCanvas(): void {
        // Enable high DPI displays
        const dpr = window.devicePixelRatio || 1;
        const rect = this.canvas.getBoundingClientRect();
        
        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;
        
        this.ctx.scale(dpr, dpr);
        this.canvas.style.width = rect.width + 'px';
        this.canvas.style.height = rect.height + 'px';
        
        this.centerX = rect.width / 2;
        this.centerY = rect.height / 3;
    }

    public clear(): void {
        this.ctx.fillStyle = '#000011';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw subtle grid
        this.drawGrid();
    }

    private drawGrid(): void {
        this.ctx.strokeStyle = '#111133';
        this.ctx.lineWidth = 0.5;
        this.ctx.globalAlpha = 0.3;

        const gridSize = 50;
        
        // Vertical lines
        for (let x = 0; x < this.canvas.width; x += gridSize) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, this.canvas.height);
            this.ctx.stroke();
        }
        
        // Horizontal lines
        for (let y = 0; y < this.canvas.height; y += gridSize) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.canvas.width, y);
            this.ctx.stroke();
        }

        this.ctx.globalAlpha = 1.0;
    }

    public drawPendulum(pendulum: DoublePendulum, options: RenderOptions = {}): void {
        const {
            alpha = 1.0,
            color = '#ffffff',
            offsetX = 0,
            offsetY = 0,
            scale = this.scale
        } = options;

        const pos = pendulum.getCartesianPositions();
        const trail = pendulum.getTrail();

        // Transform coordinates
        const pivotX = this.centerX + offsetX;
        const pivotY = this.centerY + offsetY;
        const x1 = pivotX + pos.x1 * scale;
        const y1 = pivotY + pos.y1 * scale;
        const x2 = pivotX + pos.x2 * scale;
        const y2 = pivotY + pos.y2 * scale;

        this.ctx.globalAlpha = alpha;

        // Draw trail for second bob
        if (trail.length > 1) {
            this.drawTrail(trail, pivotX, pivotY, scale, color, alpha * 0.6);
        }

        // Draw pendulum rods
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 3;
        this.ctx.lineCap = 'round';

        // First rod
        this.ctx.beginPath();
        this.ctx.moveTo(pivotX, pivotY);
        this.ctx.lineTo(x1, y1);
        this.ctx.stroke();

        // Second rod
        this.ctx.beginPath();
        this.ctx.moveTo(x1, y1);
        this.ctx.lineTo(x2, y2);
        this.ctx.stroke();

        // Draw pivot point
        this.ctx.fillStyle = '#666';
        this.ctx.beginPath();
        this.ctx.arc(pivotX, pivotY, 4, 0, Math.PI * 2);
        this.ctx.fill();

        // Draw first bob
        this.ctx.fillStyle = this.lightenColor(color, 0.8);
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.arc(x1, y1, 12, 0, Math.PI * 2);
        this.ctx.fill();
        this.ctx.stroke();

        // Draw second bob (larger, more prominent)
        this.ctx.fillStyle = color;
        this.ctx.beginPath();
        this.ctx.arc(x2, y2, 15, 0, Math.PI * 2);
        this.ctx.fill();
        this.ctx.stroke();

        // Draw velocity vectors (optional, for debugging)
        if (alpha > 0.8) {
            this.drawVelocityVectors(pendulum, pivotX, pivotY, scale, color);
        }

        this.ctx.globalAlpha = 1.0;
    }

    private drawTrail(
        trail: Array<{x1: number, y1: number, x2: number, y2: number}>,
        pivotX: number,
        pivotY: number,
        scale: number,
        color: string,
        alpha: number
    ): void {
        if (trail.length < 2) return;

        this.ctx.globalAlpha = alpha;
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 1;

        // Create gradient trail effect
        for (let i = 1; i < trail.length; i++) {
            const progress = i / trail.length;
            this.ctx.globalAlpha = alpha * progress * progress; // Quadratic fade
            
            const curr = trail[i];
            const prev = trail[i - 1];
            
            // Draw trail for second bob
            this.ctx.beginPath();
            this.ctx.moveTo(
                pivotX + prev.x2 * scale,
                pivotY + prev.y2 * scale
            );
            this.ctx.lineTo(
                pivotX + curr.x2 * scale,
                pivotY + curr.y2 * scale
            );
            this.ctx.stroke();
        }
    }

    private drawVelocityVectors(
        pendulum: DoublePendulum,
        pivotX: number,
        pivotY: number,
        scale: number,
        color: string
    ): void {
        const state = pendulum.getState();
        const pos = pendulum.getCartesianPositions();
        const params = pendulum.getParams();

        // Calculate velocity vectors
        const v1x = -params.l1 * state.omega1 * Math.cos(state.theta1);
        const v1y = -params.l1 * state.omega1 * Math.sin(state.theta1);
        const v2x = v1x - params.l2 * state.omega2 * Math.cos(state.theta2);
        const v2y = v1y - params.l2 * state.omega2 * Math.sin(state.theta2);

        const velocityScale = 10; // Scale factor for visibility

        this.ctx.strokeStyle = this.lightenColor(color, 0.5);
        this.ctx.lineWidth = 2;
        this.ctx.globalAlpha = 0.7;

        // First bob velocity vector
        const x1 = pivotX + pos.x1 * scale;
        const y1 = pivotY + pos.y1 * scale;
        this.ctx.beginPath();
        this.ctx.moveTo(x1, y1);
        this.ctx.lineTo(x1 + v1x * velocityScale, y1 + v1y * velocityScale);
        this.ctx.stroke();

        // Second bob velocity vector
        const x2 = pivotX + pos.x2 * scale;
        const y2 = pivotY + pos.y2 * scale;
        this.ctx.beginPath();
        this.ctx.moveTo(x2, y2);
        this.ctx.lineTo(x2 + v2x * velocityScale, y2 + v2y * velocityScale);
        this.ctx.stroke();
    }

    private lightenColor(color: string, factor: number): string {
        if (color.startsWith('#')) {
            const hex = color.slice(1);
            const num = parseInt(hex, 16);
            const r = Math.min(255, Math.floor((num >> 16) + (255 - (num >> 16)) * factor));
            const g = Math.min(255, Math.floor(((num >> 8) & 0x00FF) + (255 - ((num >> 8) & 0x00FF)) * factor));
            const b = Math.min(255, Math.floor((num & 0x0000FF) + (255 - (num & 0x0000FF)) * factor));
            return `rgb(${r}, ${g}, ${b})`;
        }
        return color;
    }

    public drawStats(x: number, y: number, stats: Record<string, any>): void {
        this.ctx.fillStyle = '#ffffff';
        this.ctx.font = '12px Courier New';
        this.ctx.globalAlpha = 0.8;

        let currentY = y;
        for (const [key, value] of Object.entries(stats)) {
            this.ctx.fillText(`${key}: ${value}`, x, currentY);
            currentY += 15;
        }

        this.ctx.globalAlpha = 1.0;
    }

    public resize(): void {
        this.setupCanvas();
    }

    /**
     * Draw performance visualization - population fitness distribution
     */
    public drawFitnessGraph(
        fitnessHistory: number[][],
        x: number,
        y: number,
        width: number,
        height: number
    ): void {
        if (fitnessHistory.length === 0) return;

        this.ctx.globalAlpha = 0.8;
        
        // Background
        this.ctx.fillStyle = '#222';
        this.ctx.fillRect(x, y, width, height);
        
        // Border
        this.ctx.strokeStyle = '#555';
        this.ctx.lineWidth = 1;
        this.ctx.strokeRect(x, y, width, height);

        // Find max fitness for scaling
        const maxFitness = Math.max(...fitnessHistory.flat());
        if (maxFitness === 0) return;

        // Draw fitness curves
        const colors = ['#ff4444', '#44ff44', '#4444ff', '#ffff44'];
        
        fitnessHistory.forEach((history, seriesIndex) => {
            if (history.length < 2) return;
            
            this.ctx.strokeStyle = colors[seriesIndex % colors.length];
            this.ctx.lineWidth = 2;
            this.ctx.beginPath();
            
            history.forEach((fitness, index) => {
                const plotX = x + (index / (history.length - 1)) * width;
                const plotY = y + height - (fitness / maxFitness) * height;
                
                if (index === 0) {
                    this.ctx.moveTo(plotX, plotY);
                } else {
                    this.ctx.lineTo(plotX, plotY);
                }
            });
            
            this.ctx.stroke();
        });

        this.ctx.globalAlpha = 1.0;
    }
}