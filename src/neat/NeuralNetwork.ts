export interface Connection {
    id: number;
    from: number;
    to: number;
    weight: number;
    enabled: boolean;
    innovation: number;
}

export interface Node {
    id: number;
    type: 'input' | 'hidden' | 'output';
    layer: number;
    value: number;
}

export class NeuralNetwork {
    private nodes: Map<number, Node> = new Map();
    private connections: Map<number, Connection> = new Map();
    private inputSize: number;
    private outputSize: number;

    constructor(inputSize: number, outputSize: number) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.initializeMinimalNetwork();
    }

    private initializeMinimalNetwork(): void {
        // Create input nodes
        for (let i = 0; i < this.inputSize; i++) {
            this.nodes.set(i, {
                id: i,
                type: 'input',
                layer: 0,
                value: 0
            });
        }

        // Create output nodes
        for (let i = 0; i < this.outputSize; i++) {
            const nodeId = this.inputSize + i;
            this.nodes.set(nodeId, {
                id: nodeId,
                type: 'output',
                layer: 1,
                value: 0
            });
        }

        // Create initial connections (fully connected)
        let innovationNumber = 0;
        for (let i = 0; i < this.inputSize; i++) {
            for (let j = 0; j < this.outputSize; j++) {
                const outputNodeId = this.inputSize + j;
                this.connections.set(innovationNumber, {
                    id: innovationNumber,
                    from: i,
                    to: outputNodeId,
                    weight: (Math.random() - 0.5) * 2,
                    enabled: true,
                    innovation: innovationNumber
                });
                innovationNumber++;
            }
        }
    }

    public feedForward(inputs: number[]): number[] {
        if (inputs.length !== this.inputSize) {
            throw new Error(`Expected ${this.inputSize} inputs, got ${inputs.length}`);
        }

        // Reset all node values
        for (const node of this.nodes.values()) {
            node.value = 0;
        }

        // Set input values
        for (let i = 0; i < this.inputSize; i++) {
            const node = this.nodes.get(i);
            if (node) node.value = inputs[i];
        }

        // Calculate network layers and propagate forward
        const layers = this.getLayersTopological();
        
        for (let layerIndex = 0; layerIndex < layers.length - 1; layerIndex++) {
            for (const nodeId of layers[layerIndex]) {
                const fromNode = this.nodes.get(nodeId);
                if (!fromNode) continue;

                // Propagate to connected nodes
                for (const conn of this.connections.values()) {
                    if (conn.from === nodeId && conn.enabled) {
                        const toNode = this.nodes.get(conn.to);
                        if (toNode) {
                            toNode.value += fromNode.value * conn.weight;
                        }
                    }
                }
            }

            // Apply activation function to next layer
            if (layerIndex + 1 < layers.length) {
                for (const nodeId of layers[layerIndex + 1]) {
                    const node = this.nodes.get(nodeId);
                    if (node && node.type !== 'input') {
                        node.value = this.activationFunction(node.value);
                    }
                }
            }
        }

        // Collect outputs
        const outputs: number[] = [];
        for (let i = 0; i < this.outputSize; i++) {
            const node = this.nodes.get(this.inputSize + i);
            outputs.push(node ? node.value : 0);
        }

        return outputs;
    }

    private activationFunction(x: number): number {
        // Tanh activation function
        return Math.tanh(x);
    }

    private getLayersTopological(): number[][] {
        const layers: number[][] = [];
        const visited = new Set<number>();
        const nodesByLayer = new Map<number, number[]>();

        // Group nodes by layer
        for (const node of this.nodes.values()) {
            if (!nodesByLayer.has(node.layer)) {
                nodesByLayer.set(node.layer, []);
            }
            nodesByLayer.get(node.layer)!.push(node.id);
        }

        // Sort layers by layer number
        const sortedLayers = Array.from(nodesByLayer.keys()).sort((a, b) => a - b);
        for (const layerNum of sortedLayers) {
            layers.push(nodesByLayer.get(layerNum)!);
        }

        return layers;
    }

    public addNode(fromConnection: number): number {
        const conn = this.connections.get(fromConnection);
        if (!conn || !conn.enabled) return -1;

        // Disable the old connection
        conn.enabled = false;

        // Create new hidden node
        const newNodeId = Math.max(...this.nodes.keys()) + 1;
        const fromNode = this.nodes.get(conn.from)!;
        const toNode = this.nodes.get(conn.to)!;
        
        const newLayer = Math.floor((fromNode.layer + toNode.layer) / 2);
        
        this.nodes.set(newNodeId, {
            id: newNodeId,
            type: 'hidden',
            layer: newLayer,
            value: 0
        });

        // Create two new connections
        const newInnovation1 = Math.max(...this.connections.keys()) + 1;
        const newInnovation2 = newInnovation1 + 1;

        this.connections.set(newInnovation1, {
            id: newInnovation1,
            from: conn.from,
            to: newNodeId,
            weight: 1.0,
            enabled: true,
            innovation: newInnovation1
        });

        this.connections.set(newInnovation2, {
            id: newInnovation2,
            from: newNodeId,
            to: conn.to,
            weight: conn.weight,
            enabled: true,
            innovation: newInnovation2
        });

        return newNodeId;
    }

    public addConnection(from: number, to: number): boolean {
        // Check if connection already exists
        for (const conn of this.connections.values()) {
            if (conn.from === from && conn.to === to) {
                return false; // Connection already exists
            }
        }

        // Check for cycles
        if (this.wouldCreateCycle(from, to)) {
            return false;
        }

        const newInnovation = Math.max(...this.connections.keys()) + 1;
        this.connections.set(newInnovation, {
            id: newInnovation,
            from,
            to,
            weight: (Math.random() - 0.5) * 2,
            enabled: true,
            innovation: newInnovation
        });

        return true;
    }

    private wouldCreateCycle(from: number, to: number): boolean {
        // Simple cycle detection - check if 'to' can reach 'from'
        const visited = new Set<number>();
        const stack = [to];

        while (stack.length > 0) {
            const current = stack.pop()!;
            if (current === from) return true;
            if (visited.has(current)) continue;
            
            visited.add(current);
            
            for (const conn of this.connections.values()) {
                if (conn.from === current && conn.enabled) {
                    stack.push(conn.to);
                }
            }
        }

        return false;
    }

    public mutateWeights(rate: number, strength: number): void {
        for (const conn of this.connections.values()) {
            if (Math.random() < rate) {
                if (Math.random() < 0.1) {
                    // 10% chance to completely randomize
                    conn.weight = (Math.random() - 0.5) * 2;
                } else {
                    // 90% chance to perturb
                    conn.weight += (Math.random() - 0.5) * strength;
                    conn.weight = Math.max(-5, Math.min(5, conn.weight)); // Clamp weights
                }
            }
        }
    }

    public clone(): NeuralNetwork {
        const clone = new NeuralNetwork(this.inputSize, this.outputSize);
        clone.nodes.clear();
        clone.connections.clear();

        // Copy nodes
        for (const [id, node] of this.nodes) {
            clone.nodes.set(id, { ...node });
        }

        // Copy connections
        for (const [id, conn] of this.connections) {
            clone.connections.set(id, { ...conn });
        }

        return clone;
    }

    public getComplexity(): number {
        const enabledConnections = Array.from(this.connections.values()).filter(c => c.enabled).length;
        const hiddenNodes = Array.from(this.nodes.values()).filter(n => n.type === 'hidden').length;
        return enabledConnections + hiddenNodes;
    }

    public getConnections(): Connection[] {
        return Array.from(this.connections.values());
    }

    public getNodes(): Node[] {
        return Array.from(this.nodes.values());
    }

    /**
     * Export network topology for visualization or saving
     */
    public export(): { nodes: Node[], connections: Connection[] } {
        return {
            nodes: this.getNodes(),
            connections: this.getConnections()
        };
    }

    /**
     * Import network topology
     */
    public import(data: { nodes: Node[], connections: Connection[] }): void {
        this.nodes.clear();
        this.connections.clear();

        for (const node of data.nodes) {
            this.nodes.set(node.id, { ...node });
        }

        for (const conn of data.connections) {
            this.connections.set(conn.id, { ...conn });
        }
    }
}