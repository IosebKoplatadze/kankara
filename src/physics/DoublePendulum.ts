export interface DoublePendulumState {
    theta1: number;  // Angle of first pendulum
    theta2: number;  // Angle of second pendulum
    omega1: number;  // Angular velocity of first pendulum
    omega2: number;  // Angular velocity of second pendulum
}

export interface DoublePendulumParams {
    m1: number;      // Mass of first bob
    m2: number;      // Mass of second bob
    l1: number;      // Length of first pendulum
    l2: number;      // Length of second pendulum
    g: number;       // Gravitational acceleration
    damping: number; // Damping coefficient
}

export class DoublePendulum {
    private state: DoublePendulumState;
    private params: DoublePendulumParams;
    private trail: Array<{x1: number, y1: number, x2: number, y2: number}> = [];
    private maxTrailLength: number = 500;
    private time: number = 0;
    private energy: number = 0;

    constructor(
        initialState: Partial<DoublePendulumState> = {},
        params: Partial<DoublePendulumParams> = {}
    ) {
        this.state = {
            theta1: Math.PI / 2,
            theta2: Math.PI / 2,
            omega1: 0,
            omega2: 0,
            ...initialState
        };

        this.params = {
            m1: 1.0,
            m2: 1.0,
            l1: 1.0,
            l2: 1.0,
            g: 9.81,
            damping: 0.999,
            ...params
        };

        this.calculateEnergy();
    }

    /**
     * Update the pendulum state using the Runge-Kutta 4th order method
     * for solving the coupled differential equations
     */
    public update(dt: number): void {
        const k1 = this.derivatives(this.state);
        
        const state2 = this.addStates(this.state, this.multiplyState(k1, dt * 0.5));
        const k2 = this.derivatives(state2);
        
        const state3 = this.addStates(this.state, this.multiplyState(k2, dt * 0.5));
        const k3 = this.derivatives(state3);
        
        const state4 = this.addStates(this.state, this.multiplyState(k3, dt));
        const k4 = this.derivatives(state4);

        // Combine derivatives using RK4 formula
        const finalDerivative = this.addStates(
            k1,
            this.multiplyState(k2, 2),
            this.multiplyState(k3, 2),
            k4
        );

        this.state = this.addStates(
            this.state,
            this.multiplyState(finalDerivative, dt / 6)
        );

        // Apply damping
        this.state.omega1 *= this.params.damping;
        this.state.omega2 *= this.params.damping;

        this.time += dt;
        this.calculateEnergy();
        this.updateTrail();
    }

    /**
     * Calculate the derivatives for the double pendulum system
     * Based on the Lagrangian mechanics equations
     */
    private derivatives(state: DoublePendulumState): DoublePendulumState {
        const { theta1, theta2, omega1, omega2 } = state;
        const { m1, m2, l1, l2, g } = this.params;

        const deltaTheta = theta2 - theta1;
        const cosTheta = Math.cos(deltaTheta);
        const sinTheta = Math.sin(deltaTheta);

        const denominator1 = (m1 + m2) * l1 - m2 * l1 * cosTheta * cosTheta;
        const denominator2 = (l2 / l1) * denominator1;

        // First pendulum angular acceleration
        const numerator1 = -m2 * l1 * omega1 * omega1 * sinTheta * cosTheta
                          + m2 * g * Math.sin(theta2) * cosTheta
                          + m2 * l2 * omega2 * omega2 * sinTheta
                          - (m1 + m2) * g * Math.sin(theta1);

        const alpha1 = numerator1 / denominator1;

        // Second pendulum angular acceleration
        const numerator2 = -m2 * l2 * omega2 * omega2 * sinTheta * cosTheta
                          + (m1 + m2) * g * Math.sin(theta1) * cosTheta
                          - (m1 + m2) * l1 * omega1 * omega1 * sinTheta
                          - (m1 + m2) * g * Math.sin(theta2);

        const alpha2 = numerator2 / denominator2;

        return {
            theta1: omega1,
            theta2: omega2,
            omega1: alpha1,
            omega2: alpha2
        };
    }

    private addStates(...states: DoublePendulumState[]): DoublePendulumState {
        return states.reduce((acc, state) => ({
            theta1: acc.theta1 + state.theta1,
            theta2: acc.theta2 + state.theta2,
            omega1: acc.omega1 + state.omega1,
            omega2: acc.omega2 + state.omega2
        }), { theta1: 0, theta2: 0, omega1: 0, omega2: 0 });
    }

    private multiplyState(state: DoublePendulumState, factor: number): DoublePendulumState {
        return {
            theta1: state.theta1 * factor,
            theta2: state.theta2 * factor,
            omega1: state.omega1 * factor,
            omega2: state.omega2 * factor
        };
    }

    private calculateEnergy(): void {
        const { theta1, theta2, omega1, omega2 } = this.state;
        const { m1, m2, l1, l2, g } = this.params;

        // Kinetic energy
        const ke1 = 0.5 * m1 * l1 * l1 * omega1 * omega1;
        const ke2 = 0.5 * m2 * (
            l1 * l1 * omega1 * omega1 + 
            l2 * l2 * omega2 * omega2 + 
            2 * l1 * l2 * omega1 * omega2 * Math.cos(theta1 - theta2)
        );

        // Potential energy (relative to lowest point)
        const pe1 = -m1 * g * l1 * Math.cos(theta1);
        const pe2 = -m2 * g * (l1 * Math.cos(theta1) + l2 * Math.cos(theta2));

        this.energy = ke1 + ke2 + pe1 + pe2;
    }

    private updateTrail(): void {
        const pos = this.getCartesianPositions();
        this.trail.push({
            x1: pos.x1,
            y1: pos.y1,
            x2: pos.x2,
            y2: pos.y2
        });

        if (this.trail.length > this.maxTrailLength) {
            this.trail.shift();
        }
    }

    public getCartesianPositions(): { x1: number, y1: number, x2: number, y2: number } {
        const { theta1, theta2 } = this.state;
        const { l1, l2 } = this.params;

        const x1 = l1 * Math.sin(theta1);
        const y1 = -l1 * Math.cos(theta1);
        const x2 = x1 + l2 * Math.sin(theta2);
        const y2 = y1 - l2 * Math.cos(theta2);

        return { x1, y1, x2, y2 };
    }

    public getState(): DoublePendulumState {
        return { ...this.state };
    }

    public setState(newState: Partial<DoublePendulumState>): void {
        this.state = { ...this.state, ...newState };
        this.calculateEnergy();
    }

    public getParams(): DoublePendulumParams {
        return { ...this.params };
    }

    public setParams(newParams: Partial<DoublePendulumParams>): void {
        this.params = { ...this.params, ...newParams };
    }

    public getTrail(): Array<{x1: number, y1: number, x2: number, y2: number}> {
        return [...this.trail];
    }

    public getEnergy(): number {
        return this.energy;
    }

    public getTime(): number {
        return this.time;
    }

    public reset(newState?: Partial<DoublePendulumState>): void {
        this.state = {
            theta1: Math.PI / 2,
            theta2: Math.PI / 2,
            omega1: 0,
            omega2: 0,
            ...(newState || {})
        };
        this.trail = [];
        this.time = 0;
        this.calculateEnergy();
    }

    /**
     * Apply control forces to the pendulum
     * This is used by the NEAT agents to control the system
     */
    public applyControl(force1: number, force2: number): void {
        // Add external torques to the angular accelerations
        // This would be integrated into the derivatives calculation
        // For now, we apply small impulses to the angular velocities
        this.state.omega1 += force1 * 0.01;
        this.state.omega2 += force2 * 0.01;
    }

    /**
     * Get a normalized state vector for neural network input
     */
    public getNormalizedState(): number[] {
        const { theta1, theta2, omega1, omega2 } = this.state;
        return [
            Math.sin(theta1),
            Math.cos(theta1),
            Math.sin(theta2),
            Math.cos(theta2),
            omega1 / 10, // Normalize angular velocities
            omega2 / 10,
            Math.sin(theta1 - theta2), // Relative angle
            Math.cos(theta1 - theta2)
        ];
    }

    /**
     * Calculate fitness based on how long the pendulum stays "up"
     * and how stable it is
     */
    public calculateFitness(): number {
        const pos = this.getCartesianPositions();
        const height1 = pos.y1;
        const height2 = pos.y2;
        
        // Fitness based on keeping both bobs as high as possible
        const heightFitness = (height1 + height2 + 2) * 0.5; // Normalized to [0, 1]
        
        // Bonus for stability (low angular velocities)
        const stabilityFitness = 1 / (1 + Math.abs(this.state.omega1) + Math.abs(this.state.omega2));
        
        // Time bonus - reward for lasting longer
        const timeFitness = Math.min(this.time / 10, 1); // Max bonus at 10 seconds
        
        const fitness = (heightFitness * 0.5 + stabilityFitness * 0.3 + timeFitness * 0.2) * 100;
        
        return fitness;
    }
}