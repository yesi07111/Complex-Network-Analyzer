# 🎯 Graph Analysis Tool

## 📊 Professional Graph Analysis and Visualization Suite

A powerful Python-based graph analysis tool that combines advanced network analysis capabilities with sophisticated visualization features, built with NetworkX, Matplotlib, and Tkinter. This comprehensive tool offers both analysis and simulation capabilities while maintaining high performance through intelligent caching mechanisms.

### 🌟 Key Features

#### 📈 Advanced Visualization
- **Multi-Edge Support**: Sophisticated handling of parallel edges with optimal label placement
- **Self-Loop Visualization**: Enhanced rendering of self-loops with automatic position adjustment
- **Dynamic Layout**: Automatic graph layout optimization for better readability
- **Interactive Interface**: Zoom, pan, and real-time graph manipulation
- **Custom Edge Labels**: Smart positioning system for edge weights and capacities

#### 🔍 Analysis Capabilities

##### Network Metrics
- **Centrality Measures**
  - Betweenness Centrality
  - Eigenvector Centrality
  - PageRank
  - Katz Centrality

##### Structural Analysis
- **Community Detection**
- **Rich Club Coefficient**
- **Bridge Detection**
- **Articulation Points**
- **Core Numbers**

##### Path Analysis
- **Shortest Path Algorithms**
- **Maximum Flow**
- **Global Min/Max Cost Paths**
- **Weighted Path Analysis**

##### Graph Properties
- **Clustering Coefficient**
- **Assortativity**
- **Small World Coefficient**
- **Scale-Free Testing**
- **Spectral Analysis**

#### 🎮 Information Diffusion Simulator
- **Interactive Simulation**: Real-time visualization of information spread
- **Customizable Parameters**:
  - Diffusion probabilities
  - Acceptance rates
  - Resilience coefficients
  - Information modification rates
- **Analysis Tools**:
  - Percentage of informed nodes
  - Growth rate analysis
  - Infection time distribution
- **Visualization Controls**:
  - Play/Pause functionality
  - Step-by-step navigation
  - Zoom and pan capabilities

#### 🚀 Performance Features
- **Intelligent Caching System**: Optimizes computation-heavy operations
- **Efficient Memory Management**
- **Multi-graph Support**
- **Undo/Redo Functionality**

#### 🛠 Technical Features
- **Graph Type Support**
  - Directed/Undirected
  - Weighted/Unweighted
  - Simple Graphs
  - Multigraphs
  - Pseudographs
  - Flow Networks

### 🔧 Installation

This project requires Python 3.8+ and the following libraries:
- networkx
- matplotlib
- numpy
- tkinter (usually comes with Python)

You can install the required libraries using pip:

```bash
pip install networkx matplotlib numpy
```

### 📖 Usage

To run the application:
1. Clone or download the project
2. Navigate to the project directory
3. Run:
```bash
python main.py
```

The main window will open, allowing you to:
- Create and modify graphs
- Perform various analyses
- Run information diffusion simulations
- Visualize results

### 📦 Creating Executable
You can create a standalone executable using PyInstaller:

1. Install PyInstaller:
```bash
pip install pyinstaller
```
2. Create the executable:
```bash
pyinstaller --onefile --windowed --name GraphAnalyzer main.py
```

This will create:

- A dist directory containing the standalone executable
- A build directory with build files (can be safely deleted)
- A GraphAnalyzer.spec file for build configuration
The executable can be found in the dist directory and can be run on any compatible system without requiring Python installation.

### 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
