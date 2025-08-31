import { NeuralNetwork } from './NeuralNetwork';
import { DoublePendulum } from '../physics/DoublePendulum';

export interface Individual {
    id: number;
    network: NeuralNetwork;
    fitness: number;
    pendulum?: DoublePendulum;
    species: number;
    age: number;
}

export interface Species {
    id: number;
    representative: Individual;
    members: Individual[];
    averageFitness: number;
    bestFitness: number;
    stagnantGenerations: number;
}

export interface NEATConfig {
    populationSize: number;
    inputSize: number;
    outputSize: number;
    compatibilityThreshold: number;
    c1: number; // Excess coefficient
    c2: number; // Disjoint coefficient
    c3: number; // Weight difference coefficient
    mutationRates: {
        weights: number;
        addNode: number;
        addConnection: number;
        toggleConnection: number;
    };
    survivalRate: number;
    maxStagnantGenerations: number;
}

export class NEATEvolution {
    private config: NEATConfig;
    private population: Individual[] = [];
    private species: Map<number, Species> = new Map();
    private generation: number = 0;
    private innovationNumber: number = 0;
    private nextIndividualId: number = 0;
    private nextSpeciesId: number = 0;
    private fitnessHistory: number[][] = [[], [], []]; // [best, average, worst]
    private simulationTime: number = 5000; // 5 seconds per evaluation

    constructor(populationSize: number = 50) {
        this.config = {
            populationSize,
            inputSize: 8, // DoublePendulum normalized state size
            outputSize: 2, // Two control forces
            compatibilityThreshold: 3.0,
            c1: 1.0, // Excess coefficient
            c2: 1.0, // Disjoint coefficient  
            c3: 0.4, // Weight difference coefficient
            mutationRates: {
                weights: 0.8,
                addNode: 0.03,
                addConnection: 0.05,
                toggleConnection: 0.01
            },
            survivalRate: 0.2,
            maxStagnantGenerations: 15
        };

        this.initializePopulation();
    }

    private initializePopulation(): void {
        this.population = [];
        
        for (let i = 0; i < this.config.populationSize; i++) {
            const network = new NeuralNetwork(this.config.inputSize, this.config.outputSize);
            const individual: Individual = {
                id: this.nextIndividualId++,
                network,
                fitness: 0,
                species: -1,
                age: 0
            };
            this.population.push(individual);
        }

        this.speciate();
    }

    public async update(deltaTime: number): Promise<void> {
        // Evaluate population fitness in parallel using worker pool
        await this.evaluatePopulation();
        
        // Check if generation is complete
        if (this.isGenerationComplete()) {
            this.evolveToNextGeneration();
        }
    }

    private async evaluatePopulation(): Promise<void> {
        // This would use the worker pool for parallel evaluation
        // For now, we'll simulate the evaluation locally
        for (const individual of this.population) {
            if (individual.fitness === 0) { // Not yet evaluated
                individual.fitness = await this.evaluateIndividual(individual);
            }
        }
    }

    private async evaluateIndividual(individual: Individual): Promise<number> {
        // Create a pendulum for this individual
        const pendulum = new DoublePendulum({
            theta1: Math.PI * 0.8 + (Math.random() - 0.5) * 0.4,
            theta2: Math.PI * 0.8 + (Math.random() - 0.5) * 0.4,
            omega1: 0,
            omega2: 0
        });

        individual.pendulum = pendulum;

        let totalFitness = 0;
        const dt = 1 / 60; // 60 FPS
        const steps = Math.floor(this.simulationTime / 1000 * 60);

        for (let step = 0; step < steps; step++) {
            // Get normalized state for neural network
            const inputs = pendulum.getNormalizedState();
            
            // Get control outputs from neural network
            const outputs = individual.network.feedForward(inputs);
            
            // Apply control forces
            pendulum.applyControl(outputs[0], outputs[1]);
            
            // Update physics
            pendulum.update(dt);
            
            // Accumulate fitness
            totalFitness += pendulum.calculateFitness();
            
            // Early termination if pendulum falls completely
            const pos = pendulum.getCartesianPositions();
            if (pos.y1 < -2.5 || pos.y2 < -2.5) {
                break;
            }
        }

        return totalFitness / steps;
    }

    private isGenerationComplete(): boolean {
        return this.population.every(individual => individual.fitness > 0);
    }

    private evolveToNextGeneration(): void {
        // Update species fitness and remove stagnant species
        this.updateSpeciesFitness();
        this.removeStagnantSpecies();
        
        // Calculate adjusted fitness (fitness sharing)
        this.calculateAdjustedFitness();
        
        // Sort population by adjusted fitness
        this.population.sort((a, b) => b.fitness - a.fitness);
        
        // Track fitness history
        this.updateFitnessHistory();
        
        // Create new generation
        const newPopulation = this.createNewGeneration();
        
        // Replace population
        this.population = newPopulation;
        this.generation++;
        
        // Re-speciate
        this.speciate();
        
        console.log(`Generation ${this.generation} complete. Best fitness: ${this.getBestFitness().toFixed(2)}`);
    }

    private updateSpeciesFitness(): void {
        for (const species of this.species.values()) {
            const fitnesses = species.members.map(m => m.fitness);
            species.averageFitness = fitnesses.reduce((a, b) => a + b, 0) / fitnesses.length;
            species.bestFitness = Math.max(...fitnesses);
            
            // Check for improvement
            if (species.bestFitness <= species.bestFitness) {
                species.stagnantGenerations++;
            } else {
                species.stagnantGenerations = 0;
            }
        }
    }

    private removeStagnantSpecies(): void {
        const speciesToRemove: number[] = [];
        
        for (const [id, species] of this.species) {
            if (species.stagnantGenerations >= this.config.maxStagnantGenerations) {
                speciesToRemove.push(id);
            }
        }
        
        for (const id of speciesToRemove) {
            this.species.delete(id);
        }
    }

    private calculateAdjustedFitness(): void {
        for (const individual of this.population) {
            const speciesSize = this.species.get(individual.species)?.members.length || 1;
            individual.fitness = individual.fitness / speciesSize; // Fitness sharing
        }
    }

    private createNewGeneration(): Individual[] {
        const newPopulation: Individual[] = [];
        const eliteCount = Math.max(1, Math.floor(this.config.populationSize * 0.1));
        
        // Keep elite individuals
        for (let i = 0; i < eliteCount; i++) {
            if (i < this.population.length) {
                const clone = this.cloneIndividual(this.population[i]);
                clone.age++;
                newPopulation.push(clone);
            }
        }

        // Fill rest with offspring
        while (newPopulation.length < this.config.populationSize) {
            // Select parents using tournament selection
            const parent1 = this.tournamentSelection();
            const parent2 = this.tournamentSelection();
            
            // Create offspring
            const offspring = this.crossover(parent1, parent2);
            this.mutate(offspring);
            
            newPopulation.push(offspring);
        }

        return newPopulation;
    }

    private tournamentSelection(tournamentSize: number = 3): Individual {
        const tournament: Individual[] = [];
        
        for (let i = 0; i < tournamentSize; i++) {
            const randomIndex = Math.floor(Math.random() * this.population.length);
            tournament.push(this.population[randomIndex]);
        }
        
        return tournament.reduce((best, current) => 
            current.fitness > best.fitness ? current : best
        );
    }

    private crossover(parent1: Individual, parent2: Individual): Individual {
        // NEAT crossover - align by innovation numbers
        const offspring = new NeuralNetwork(this.config.inputSize, this.config.outputSize);
        
        // For simplicity, we'll take the better parent's structure and randomly mix weights
        const betterParent = parent1.fitness >= parent2.fitness ? parent1 : parent2;
        const worseParent = parent1.fitness >= parent2.fitness ? parent2 : parent1;
        
        // Clone the better parent and mix in some traits from worse parent
        const clonedNetwork = betterParent.network.clone();
        
        // Randomly mix weights from both parents
        for (const conn of clonedNetwork.getConnections()) {
            if (Math.random() < 0.5) {
                // Try to find matching connection in other parent
                const otherConn = worseParent.network.getConnections().find(c => 
                    c.from === conn.from && c.to === conn.to
                );
                if (otherConn) {
                    conn.weight = otherConn.weight;
                }
            }
        }

        return {
            id: this.nextIndividualId++,
            network: clonedNetwork,
            fitness: 0,
            species: -1,
            age: 0
        };
    }

    private mutate(individual: Individual): void {
        const rates = this.config.mutationRates;
        
        // Mutate weights
        individual.network.mutateWeights(rates.weights, 0.1);
        
        // Add node mutation
        if (Math.random() < rates.addNode) {
            const connections = individual.network.getConnections().filter(c => c.enabled);
            if (connections.length > 0) {
                const randomConn = connections[Math.floor(Math.random() * connections.length)];
                individual.network.addNode(randomConn.id);
            }
        }
        
        // Add connection mutation
        if (Math.random() < rates.addConnection) {
            const nodes = individual.network.getNodes();
            if (nodes.length >= 2) {
                const from = nodes[Math.floor(Math.random() * nodes.length)];
                const to = nodes[Math.floor(Math.random() * nodes.length)];
                individual.network.addConnection(from.id, to.id);
            }
        }
    }

    private speciate(): void {
        // Clear species memberships
        for (const species of this.species.values()) {
            species.members = [];
        }

        // Assign individuals to species
        for (const individual of this.population) {
            let assignedSpecies: Species | null = null;
            
            // Try to assign to existing species
            for (const species of this.species.values()) {
                const distance = this.calculateCompatibilityDistance(
                    individual, 
                    species.representative
                );
                
                if (distance < this.config.compatibilityThreshold) {
                    assignedSpecies = species;
                    break;
                }
            }
            
            // Create new species if no compatible species found
            if (!assignedSpecies) {
                assignedSpecies = {
                    id: this.nextSpeciesId++,
                    representative: individual,
                    members: [],
                    averageFitness: 0,
                    bestFitness: 0,
                    stagnantGenerations: 0
                };
                this.species.set(assignedSpecies.id, assignedSpecies);
            }
            
            individual.species = assignedSpecies.id;
            assignedSpecies.members.push(individual);
        }

        // Remove empty species
        const emptySpecies: number[] = [];
        for (const [id, species] of this.species) {
            if (species.members.length === 0) {
                emptySpecies.push(id);
            }
        }
        for (const id of emptySpecies) {
            this.species.delete(id);
        }
    }

    private calculateCompatibilityDistance(ind1: Individual, ind2: Individual): number {
        const conn1 = ind1.network.getConnections();
        const conn2 = ind2.network.getConnections();
        
        const innovations1 = new Set(conn1.map(c => c.innovation));
        const innovations2 = new Set(conn2.map(c => c.innovation));
        
        let excess = 0;
        let disjoint = 0;
        let weightDiff = 0;
        let matching = 0;

        const maxInnovation1 = Math.max(...innovations1);
        const maxInnovation2 = Math.max(...innovations2);
        const maxInnovation = Math.max(maxInnovation1, maxInnovation2);

        // Count excess, disjoint, and matching genes
        for (const innovation of new Set([...innovations1, ...innovations2])) {
            const in1 = innovations1.has(innovation);
            const in2 = innovations2.has(innovation);
            
            if (in1 && in2) {
                // Matching gene
                matching++;
                const weight1 = conn1.find(c => c.innovation === innovation)!.weight;
                const weight2 = conn2.find(c => c.innovation === innovation)!.weight;
                weightDiff += Math.abs(weight1 - weight2);
            } else if (innovation > Math.min(maxInnovation1, maxInnovation2)) {
                // Excess gene
                excess++;
            } else {
                // Disjoint gene
                disjoint++;
            }
        }

        const N = Math.max(conn1.length, conn2.length, 1);
        const avgWeightDiff = matching > 0 ? weightDiff / matching : 0;

        return (this.config.c1 * excess / N) + 
               (this.config.c2 * disjoint / N) + 
               (this.config.c3 * avgWeightDiff);
    }

    private cloneIndividual(individual: Individual): Individual {
        return {
            id: this.nextIndividualId++,
            network: individual.network.clone(),
            fitness: 0,
            species: -1,
            age: individual.age
        };
    }

    private updateFitnessHistory(): void {
        const fitnesses = this.population.map(ind => ind.fitness);
        this.fitnessHistory[0].push(Math.max(...fitnesses)); // Best
        this.fitnessHistory[1].push(fitnesses.reduce((a, b) => a + b, 0) / fitnesses.length); // Average
        this.fitnessHistory[2].push(Math.min(...fitnesses)); // Worst
        
        // Keep only last 100 generations
        if (this.fitnessHistory[0].length > 100) {
            this.fitnessHistory.forEach(history => history.shift());
        }
    }

    // Public API methods
    public getBestIndividual(): Individual | null {
        if (this.population.length === 0) return null;
        return this.population.reduce((best, current) => 
            current.fitness > best.fitness ? current : best
        );
    }

    public getBestIndividuals(count: number): Individual[] {
        return [...this.population]
            .sort((a, b) => b.fitness - a.fitness)
            .slice(0, count);
    }

    public getPopulation(): Individual[] {
        return [...this.population];
    }

    public getGeneration(): number {
        return this.generation;
    }

    public getBestFitness(): number {
        const best = this.getBestIndividual();
        return best ? best.fitness : 0;
    }

    public getAverageFitness(): number {
        if (this.population.length === 0) return 0;
        const total = this.population.reduce((sum, ind) => sum + ind.fitness, 0);
        return total / this.population.length;
    }

    public getMaxFitness(): number {
        return Math.max(...this.population.map(ind => ind.fitness));
    }

    public getSpeciesCount(): number {
        return this.species.size;
    }

    public getFitnessHistory(): number[][] {
        return this.fitnessHistory.map(history => [...history]);
    }

    public setPopulationSize(size: number): void {
        this.config.populationSize = size;
        if (this.population.length === 0) {
            this.initializePopulation();
        }
    }

    public reset(): void {
        this.generation = 0;
        this.population = [];
        this.species.clear();
        this.fitnessHistory = [[], [], []];
        this.nextIndividualId = 0;
        this.nextSpeciesId = 0;
        this.initializePopulation();
    }

    /**
     * Export the current best individual for saving or analysis
     */
    public exportBest(): any {
        const best = this.getBestIndividual();
        if (!best) return null;
        
        return {
            generation: this.generation,
            fitness: best.fitness,
            network: best.network.export(),
            complexity: best.network.getComplexity()
        };
    }

    /**
     * Import a previously evolved individual
     */
    public importIndividual(data: any): void {
        const network = new NeuralNetwork(this.config.inputSize, this.config.outputSize);
        network.import(data.network);
        
        const individual: Individual = {
            id: this.nextIndividualId++,
            network,
            fitness: data.fitness || 0,
            species: -1,
            age: 0
        };
        
        this.population.push(individual);
        this.speciate();
    }
}