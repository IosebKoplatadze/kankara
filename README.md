# Kankara - Sophisticated Double Pendulum NEAT Simulation

A high-performance, browser-based double pendulum simulation featuring NEAT (NeuroEvolution of Augmenting Topologies) algorithm training with parallelized computation.

![Double Pendulum Simulation](https://github.com/user-attachments/assets/8085bb09-6b6f-4e01-81ea-eb95579239b6)

## Features

### 🎯 Core Physics
- **Accurate Double Pendulum Simulation**: Implements the full Lagrangian mechanics equations using Runge-Kutta 4th order integration
- **Real-time Performance**: Optimized for 60 FPS smooth animation
- **Chaotic Dynamics**: Demonstrates sensitive dependence on initial conditions
- **Energy Conservation**: Tracks kinetic and potential energy with damping

### 🧠 AI & Machine Learning
- **NEAT Algorithm**: Complete implementation of NeuroEvolution of Augmenting Topologies
- **Neural Network Evolution**: Dynamic topology evolution with node and connection mutations
- **Species Management**: Automatic speciation and fitness sharing
- **Fitness Evaluation**: Multi-component fitness function rewarding height, stability, and longevity

### ⚡ Performance & Parallelization
- **Web Workers**: Multi-threaded simulation evaluation for enhanced performance
- **TypeScript**: Type-safe, maintainable codebase
- **Modern Build System**: Vite-powered development with hot module replacement
- **Optimized Rendering**: High-DPI support with efficient Canvas 2D rendering

### 🎨 Visualization
- **Beautiful Trail Rendering**: Gradient trail visualization showing pendulum trajectory
- **Multiple View Modes**: Best individual, population, and side-by-side comparison
- **Real-time Statistics**: FPS, generation, fitness metrics, and species count
- **Interactive Controls**: Population size, simulation speed, and view mode controls

## Getting Started

### Prerequisites
- Node.js 16+ 
- Modern web browser with Web Workers support

### Installation
```bash
npm install
```

### Development
```bash
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.

### Production Build
```bash
npm run build
npm run preview
```

## Usage

1. **Start Simulation**: Click "Start Evolution" to begin the physics simulation
2. **Pause/Resume**: Use "Pause" to stop/resume the simulation
3. **Reset**: Click "Reset" to randomize pendulum initial conditions
4. **Adjust Settings**: 
   - Population Size: Control the number of neural networks (10-500)
   - Simulation Speed: Adjust the time scale (0.1x - 5x)
   - View Mode: Switch between different visualization modes

## Technical Architecture

### Physics Engine (`src/physics/DoublePendulum.ts`)
- Implements coupled differential equations for double pendulum dynamics
- Uses Runge-Kutta 4th order method for numerical integration
- Supports external control forces for AI agent interaction
- Provides normalized state vectors for neural network input

### NEAT Implementation (`src/neat/`)
- **NeuralNetwork.ts**: Feed-forward neural networks with dynamic topology
- **NEATEvolution.ts**: Population management, speciation, and evolution

### Visualization (`src/visualization/Renderer.ts`)
- High-performance Canvas 2D rendering
- Trail visualization with gradient effects
- Multiple rendering modes for population visualization
- Performance optimizations for smooth 60 FPS

### Parallelization (`src/workers/`)
- **WorkerPool.ts**: Manages multiple Web Workers for parallel simulation
- **SimulationWorker.ts**: Individual worker for fitness evaluation
- Batch processing for efficient population evaluation

## Advanced Features

### Neural Network Architecture
- **Input Layer**: 8 neurons (pendulum state: sin/cos of angles, angular velocities, relative measurements)
- **Output Layer**: 2 neurons (control forces for each pendulum joint)
- **Dynamic Topology**: Networks evolve complexity through node and connection mutations

### Fitness Function
Multi-component fitness evaluation:
- **Height Component** (50%): Rewards keeping both pendulum masses high
- **Stability Component** (30%): Rewards low angular velocities
- **Time Component** (20%): Rewards longer survival times
- **Complexity Penalty**: Encourages simpler, more efficient solutions

### Evolution Parameters
- Population size: Configurable (default: 50)
- Mutation rates: Optimized for pendulum control task
- Species threshold: Automatic diversity maintenance
- Stagnation detection: Removes non-improving species

## Performance Optimizations

- **Efficient Physics**: Optimized differential equation solving
- **Smart Rendering**: Only renders necessary elements
- **Memory Management**: Bounded trail lengths and history buffers
- **Parallel Evaluation**: Utilizes all available CPU cores
- **Type Safety**: TypeScript prevents runtime errors

## Future Enhancements

- [ ] WebGL rendering for larger populations
- [ ] Real-time network topology visualization
- [ ] Parameter sweeps and hyperparameter optimization
- [ ] Save/load evolved networks
- [ ] Competition mode between multiple agents
- [ ] Physics parameter variation (gravity, damping, mass ratios)

## Contributing

This is a demonstration of sophisticated web-based physics simulation and AI training. Feel free to extend the codebase with additional features or optimizations.

## License

MIT License - see [LICENSE](LICENSE) file for details.
