/**
 * Mathematical utilities for the double pendulum simulation
 */

export class MathUtils {
    /**
     * Normalize angle to [-π, π]
     */
    static normalizeAngle(angle: number): number {
        while (angle > Math.PI) angle -= 2 * Math.PI;
        while (angle < -Math.PI) angle += 2 * Math.PI;
        return angle;
    }

    /**
     * Linear interpolation
     */
    static lerp(a: number, b: number, t: number): number {
        return a + (b - a) * t;
    }

    /**
     * Clamp value between min and max
     */
    static clamp(value: number, min: number, max: number): number {
        return Math.max(min, Math.min(max, value));
    }

    /**
     * Map value from one range to another
     */
    static map(value: number, fromMin: number, fromMax: number, toMin: number, toMax: number): number {
        const normalized = (value - fromMin) / (fromMax - fromMin);
        return toMin + normalized * (toMax - toMin);
    }

    /**
     * Calculate distance between two points
     */
    static distance(x1: number, y1: number, x2: number, y2: number): number {
        const dx = x2 - x1;
        const dy = y2 - y1;
        return Math.sqrt(dx * dx + dy * dy);
    }

    /**
     * Generate random number with normal distribution (Box-Muller transform)
     */
    static randomNormal(mean: number = 0, stdDev: number = 1): number {
        let u = 0, v = 0;
        while (u === 0) u = Math.random(); // Converting [0,1) to (0,1)
        while (v === 0) v = Math.random();
        
        const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
        return z * stdDev + mean;
    }

    /**
     * Calculate exponential moving average
     */
    static exponentialMovingAverage(newValue: number, currentAverage: number, alpha: number): number {
        return alpha * newValue + (1 - alpha) * currentAverage;
    }
}

/**
 * Performance monitoring utilities
 */
export class PerformanceMonitor {
    private measurements: Map<string, number[]> = new Map();
    private startTimes: Map<string, number> = new Map();

    public start(label: string): void {
        this.startTimes.set(label, performance.now());
    }

    public end(label: string): number {
        const startTime = this.startTimes.get(label);
        if (startTime === undefined) return 0;

        const duration = performance.now() - startTime;
        this.startTimes.delete(label);

        if (!this.measurements.has(label)) {
            this.measurements.set(label, []);
        }
        
        const measurements = this.measurements.get(label)!;
        measurements.push(duration);
        
        // Keep only last 100 measurements
        if (measurements.length > 100) {
            measurements.shift();
        }

        return duration;
    }

    public getAverage(label: string): number {
        const measurements = this.measurements.get(label);
        if (!measurements || measurements.length === 0) return 0;
        
        return measurements.reduce((sum, val) => sum + val, 0) / measurements.length;
    }

    public getStats(label: string): { avg: number, min: number, max: number, count: number } {
        const measurements = this.measurements.get(label);
        if (!measurements || measurements.length === 0) {
            return { avg: 0, min: 0, max: 0, count: 0 };
        }

        return {
            avg: this.getAverage(label),
            min: Math.min(...measurements),
            max: Math.max(...measurements),
            count: measurements.length
        };
    }

    public getAllStats(): Record<string, { avg: number, min: number, max: number, count: number }> {
        const stats: Record<string, any> = {};
        for (const label of this.measurements.keys()) {
            stats[label] = this.getStats(label);
        }
        return stats;
    }

    public clear(label?: string): void {
        if (label) {
            this.measurements.delete(label);
            this.startTimes.delete(label);
        } else {
            this.measurements.clear();
            this.startTimes.clear();
        }
    }
}

/**
 * Color utilities for visualization
 */
export class ColorUtils {
    /**
     * Convert HSL to RGB
     */
    static hslToRgb(h: number, s: number, l: number): [number, number, number] {
        h /= 360;
        s /= 100;
        l /= 100;

        const a = s * Math.min(l, 1 - l);
        const f = (n: number) => {
            const k = (n + h * 12) % 12;
            return l - a * Math.max(Math.min(k - 3, 9 - k, 1), -1);
        };
        
        return [Math.round(f(0) * 255), Math.round(f(8) * 255), Math.round(f(4) * 255)];
    }

    /**
     * Generate color based on fitness value
     */
    static fitnessToColor(fitness: number, maxFitness: number): string {
        const normalizedFitness = Math.max(0, Math.min(1, fitness / maxFitness));
        
        // Red (0) to Yellow (60) to Green (120) color gradient
        const hue = normalizedFitness * 120;
        const saturation = 80;
        const lightness = 40 + normalizedFitness * 20;
        
        const [r, g, b] = this.hslToRgb(hue, saturation, lightness);
        return `rgb(${r}, ${g}, ${b})`;
    }

    /**
     * Generate distinct colors for species visualization
     */
    static generateSpeciesColors(count: number): string[] {
        const colors: string[] = [];
        const goldenRatio = 0.618033988749;
        let hue = Math.random();
        
        for (let i = 0; i < count; i++) {
            hue += goldenRatio;
            hue %= 1;
            
            const [r, g, b] = this.hslToRgb(hue * 360, 70, 50);
            colors.push(`rgb(${r}, ${g}, ${b})`);
        }
        
        return colors;
    }
}

/**
 * Statistical utilities for analyzing evolution progress
 */
export class StatsUtils {
    /**
     * Calculate standard deviation
     */
    static standardDeviation(values: number[]): number {
        if (values.length === 0) return 0;
        
        const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
        const squaredDifferences = values.map(val => Math.pow(val - mean, 2));
        const variance = squaredDifferences.reduce((sum, val) => sum + val, 0) / values.length;
        
        return Math.sqrt(variance);
    }

    /**
     * Calculate percentiles
     */
    static percentile(values: number[], p: number): number {
        if (values.length === 0) return 0;
        
        const sorted = [...values].sort((a, b) => a - b);
        const index = (p / 100) * (sorted.length - 1);
        
        if (Number.isInteger(index)) {
            return sorted[index];
        } else {
            const lower = sorted[Math.floor(index)];
            const upper = sorted[Math.ceil(index)];
            return lower + (upper - lower) * (index - Math.floor(index));
        }
    }

    /**
     * Calculate diversity metrics for population
     */
    static calculateDiversity(fitnesses: number[]): {
        mean: number;
        std: number;
        range: number;
        q25: number;
        q50: number;
        q75: number;
    } {
        if (fitnesses.length === 0) {
            return { mean: 0, std: 0, range: 0, q25: 0, q50: 0, q75: 0 };
        }

        const mean = fitnesses.reduce((sum, f) => sum + f, 0) / fitnesses.length;
        const std = this.standardDeviation(fitnesses);
        const min = Math.min(...fitnesses);
        const max = Math.max(...fitnesses);
        
        return {
            mean,
            std,
            range: max - min,
            q25: this.percentile(fitnesses, 25),
            q50: this.percentile(fitnesses, 50),
            q75: this.percentile(fitnesses, 75)
        };
    }
}

/**
 * Local storage utilities for saving/loading evolution progress
 */
export class StorageUtils {
    private static readonly STORAGE_KEY = 'kankara_evolution_data';

    /**
     * Save evolution state to local storage
     */
    static saveEvolutionState(data: any): void {
        try {
            const serialized = JSON.stringify(data);
            localStorage.setItem(this.STORAGE_KEY, serialized);
        } catch (error) {
            console.warn('Failed to save evolution state:', error);
        }
    }

    /**
     * Load evolution state from local storage
     */
    static loadEvolutionState(): any | null {
        try {
            const serialized = localStorage.getItem(this.STORAGE_KEY);
            return serialized ? JSON.parse(serialized) : null;
        } catch (error) {
            console.warn('Failed to load evolution state:', error);
            return null;
        }
    }

    /**
     * Clear saved evolution state
     */
    static clearEvolutionState(): void {
        localStorage.removeItem(this.STORAGE_KEY);
    }

    /**
     * Export data as downloadable file
     */
    static exportToFile(data: any, filename: string): void {
        const blob = new Blob([JSON.stringify(data, null, 2)], { 
            type: 'application/json' 
        });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
}