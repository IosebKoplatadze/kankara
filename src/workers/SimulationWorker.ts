import { DoublePendulum, DoublePendulumState } from '../physics/DoublePendulum';
import { NeuralNetwork } from '../neat/NeuralNetwork';

export interface WorkerTask {
    id: string;
    individual: {
        id: number;
        network: any; // Serialized network data
        fitness: number;
        species: number;
        age: number;
    };
    simulationTime: number;
}

export interface WorkerResult {
    taskId: string;
    fitness: number;
    simulationData?: any;
}

// Worker script for parallel simulation evaluation
self.onmessage = (event: MessageEvent) => {
    const { type, task } = event.data;

    if (type === 'evaluate') {
        try {
            const result = evaluateIndividual(task);
            self.postMessage({
                type: 'result',
                result
            });
        } catch (error) {
            self.postMessage({
                type: 'error',
                error: {
                    message: error instanceof Error ? error.message : 'Unknown error',
                    taskId: task.id
                }
            });
        }
    }
};

function evaluateIndividual(task: WorkerTask): WorkerResult {
    const { individual, simulationTime } = task;
    
    // Recreate neural network from exported data
    const network = new NeuralNetwork(8, 2); // 8 inputs, 2 outputs
    network.import(individual.network); // individual.network is serialized data
    
    // Create pendulum with slight randomization
    const pendulum = new DoublePendulum({
        theta1: Math.PI * 0.8 + (Math.random() - 0.5) * 0.4,
        theta2: Math.PI * 0.8 + (Math.random() - 0.5) * 0.4,
        omega1: (Math.random() - 0.5) * 0.1,
        omega2: (Math.random() - 0.5) * 0.1
    });

    let totalFitness = 0;
    let maxHeight = 0;
    let stabilityScore = 0;
    const dt = 1 / 60; // 60 FPS simulation
    const steps = Math.floor(simulationTime / 1000 * 60);
    
    const simulationData = {
        trajectory: [] as Array<{x1: number, y1: number, x2: number, y2: number, time: number}>,
        energyHistory: [] as number[],
        controlHistory: [] as Array<{force1: number, force2: number}>
    };

    for (let step = 0; step < steps; step++) {
        // Get current state for neural network
        const inputs = pendulum.getNormalizedState();
        
        // Get control outputs from neural network
        const outputs = network.feedForward(inputs);
        
        // Apply smooth control (tanh to bound outputs)
        const force1 = Math.tanh(outputs[0]) * 0.5;
        const force2 = Math.tanh(outputs[1]) * 0.5;
        
        pendulum.applyControl(force1, force2);
        
        // Update physics
        pendulum.update(dt);
        
        // Calculate instantaneous fitness
        const instantFitness = calculateInstantaneousFitness(pendulum, step * dt);
        totalFitness += instantFitness;
        
        // Track metrics
        const pos = pendulum.getCartesianPositions();
        const currentHeight = pos.y1 + pos.y2;
        maxHeight = Math.max(maxHeight, currentHeight);
        
        // Stability: reward low angular velocities when pendulum is upright
        const state = pendulum.getState();
        if (pos.y1 > -0.5 && pos.y2 > -0.5) { // Both bobs relatively high
            const angularVelocityMagnitude = Math.abs(state.omega1) + Math.abs(state.omega2);
            stabilityScore += 1 / (1 + angularVelocityMagnitude);
        }
        
        // Record simulation data (sample every 10 steps to reduce memory)
        if (step % 10 === 0) {
            simulationData.trajectory.push({
                x1: pos.x1,
                y1: pos.y1,
                x2: pos.x2,
                y2: pos.y2,
                time: step * dt
            });
            simulationData.energyHistory.push(pendulum.getEnergy());
            simulationData.controlHistory.push({ force1, force2 });
        }
        
        // Early termination if pendulum fails catastrophically
        if (pos.y1 < -3.0 && pos.y2 < -3.0) {
            // Penalty for early failure
            totalFitness *= (step / steps);
            break;
        }
    }

    // Calculate final fitness with bonuses
    let finalFitness = totalFitness / steps;
    
    // Height bonus
    const heightBonus = Math.max(0, maxHeight + 2) * 10; // Bonus for keeping pendulum high
    
    // Stability bonus
    const stabilityBonus = (stabilityScore / steps) * 20;
    
    // Complexity penalty (encourage simpler solutions)
    const complexity = network.getComplexity();
    const complexityPenalty = complexity * 0.1;
    
    finalFitness = finalFitness + heightBonus + stabilityBonus - complexityPenalty;
    
    return {
        taskId: task.id,
        fitness: Math.max(0, finalFitness), // Ensure non-negative fitness
        simulationData
    };
}

function calculateInstantaneousFitness(pendulum: DoublePendulum, time: number): number {
    const pos = pendulum.getCartesianPositions();
    const state = pendulum.getState();
    
    // Height component - reward keeping both bobs high
    const heightFitness = (pos.y1 + pos.y2 + 4) * 0.25; // Normalized
    
    // Balance component - reward balanced positions
    const balanceFitness = 1 / (1 + Math.abs(pos.x1) + Math.abs(pos.x2));
    
    // Stability component - reward low angular velocities
    const stabilityFitness = 1 / (1 + Math.abs(state.omega1) + Math.abs(state.omega2));
    
    // Energy efficiency - penalize extreme energy levels
    const energy = pendulum.getEnergy();
    const energyFitness = 1 / (1 + Math.abs(energy) * 0.1);
    
    // Time bonus - encourage longer survival
    const timeBonus = Math.min(time / 10, 1.0);
    
    return (
        heightFitness * 0.4 +
        balanceFitness * 0.2 +
        stabilityFitness * 0.2 +
        energyFitness * 0.1 +
        timeBonus * 0.1
    );
}

// Export for main thread if needed
export { };