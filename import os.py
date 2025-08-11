import os
import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')

class FrictionalContactAnalyzer:
    """
    Analyze frictional contact metrics (Zi, P(Zi), <Z>, <Z_net>) for 3D suspension simulation data
    """
    
    def __init__(self, data_dir, phi, ar, vr, run_num=1):
        """Initialize the analyzer with simulation parameters"""
        self.data_dir = Path(data_dir)
        self.phi = phi
        self.ar = ar
        self.vr = vr
        self.run_num = run_num
        
        # Data storage
        self.bulk_data = {}
        self.adjacency_matrices = []
        self.network_metrics = {}
        
        print(f"Initialized frictional contact analyzer for φ={phi}, ar={ar}, vr={vr}")
    
    def extract_metadata(self, file_path):
        """Extract number of particles from header"""
        with open(file_path, 'r') as f:
            header_lines = [line.strip() for line in f.readlines() if line.startswith('#')][:22]
        
        npp = None
        for line in header_lines:
            if line.startswith('# np'):
                npp = int(line.split()[-1])
        
        if npp is None:
            raise ValueError(f"Could not extract number of particles from {file_path}")
        
        return npp
    
    def read_data_file(self, file_path):
        """Read bulk data file for shear rate"""
        data = np.loadtxt(file_path)
        return {'shear_rate': data[:, 2]}
    
    def read_interaction_file(self, file_path):
        """Read interaction/contact data"""
        with open(file_path, 'r') as f:
            lines = f.readlines()[27:]  # Skip 27 header lines
        
        interactions = []
        temp = []
        hashCounter = 0
        
        for line in lines:
            if line.startswith('#'):
                hashCounter += 1
                if hashCounter == 7 and temp:
                    if temp:
                        interactions.append(np.array(temp))
                    temp = []
                    hashCounter = 0
            else:
                temp.append([float(x) for x in line.split()])
        
        if temp:
            interactions.append(np.array(temp))
        
        return interactions
    
    def load_simulation_data(self):
        """Load simulation data files from phi_X.XX/ar_Y.Y/Vr_Z.Z/run_N"""
        phi_str = '{:.3f}'.format(self.phi) if len(str(self.phi).split('.')[1]) > 2 else '{:.2f}'.format(self.phi)
        dataname = self.data_dir / f"phi_{phi_str}/ar_{self.ar}/Vr_{self.vr}/run_{self.run_num}"
        
        print(f"Looking for files in: {dataname}")
        if not os.path.exists(dataname):
            raise FileNotFoundError(f"Directory not found: {dataname}")
        
        # Find files
        par_files = sorted(glob.glob(f'{dataname}/par_*.dat'))
        data_files = sorted(glob.glob(f'{dataname}/data_*.dat'))
        interaction_files = sorted(glob.glob(f'{dataname}/int_*.dat'))
        
        print(f"Found par_files: {par_files}")
        print(f"Found data_files: {data_files}")
        print(f"Found interaction_files: {interaction_files}")
        
        if not (par_files and data_files and interaction_files):
            raise FileNotFoundError(f"Missing required data files in {dataname}")
        
        # Process first matching set
        for par_file in par_files:
            base_name = os.path.basename(par_file).replace('par_', '').replace('.dat', '')
            data_file = next((f for f in data_files if base_name in f), None)
            interaction_file = next((f for f in interaction_files if base_name in f), None)
            
            if not (data_file and interaction_file):
                continue
            
            print(f"Loading data from {base_name}...")
            
            # Extract metadata
            self.npp = self.extract_metadata(par_file)
            
            # Load data
            self.bulk_data = self.read_data_file(data_file)
            interaction_data = self.read_interaction_file(interaction_file)
            
            # Process interactions to get frictional contacts
            self.process_interactions(interaction_data)
            
            break  # Process first matching set
        
        print(f"Loaded {len(self.adjacency_matrices)} timesteps with {self.npp} particles")
        return True
    
    def process_interactions(self, interaction_data):
        """Process interaction data to extract frictional contact networks"""
        self.adjacency_matrices = []
        
        for frame in interaction_data:
            if len(frame) == 0:
                # Empty frame
                self.adjacency_matrices.append(np.zeros((self.npp, self.npp)))
                continue
            
            # Extract data
            particle_pairs = frame[:, [0, 1]].astype(int)
            contact_states = frame[:, 10]  # Contact state (1 for frictional, 0 for frictionless)
            
            # Extract frictional contacts only (contact_state == 1)
            frictional_mask = contact_states == 1
            frictional_contacts = particle_pairs[frictional_mask]
            
            # Build adjacency matrix for frictional contacts
            adj_matrix = np.zeros((self.npp, self.npp))
            if len(frictional_contacts) > 0:
                for pair in frictional_contacts:
                    i, j = int(pair[0]), int(pair[1])
                    if 0 <= i < self.npp and 0 <= j < self.npp:
                        adj_matrix[i, j] = 1
                        adj_matrix[j, i] = 1  # Symmetric for undirected graph
            
            self.adjacency_matrices.append(adj_matrix)
    
    def calculate_degree_distribution(self):
        """Calculate degree distribution P(Z_i), <Z>, and <Z_net> for frictional contacts"""
        all_degrees = []
        degree_distributions = []
        mean_degree = np.zeros(len(self.adjacency_matrices))
        net_degree = np.zeros(len(self.adjacency_matrices))
        
        for t, adj_matrix in enumerate(self.adjacency_matrices):
            degrees = np.sum(adj_matrix, axis=1)  # Z_i for each particle
            all_degrees.extend(degrees)
            degree_distributions.append(degrees)
            
            # Mean degree <Z>
            mean_degree[t] = np.mean(degrees)
            
            # Net degree <Z_net> (mean degree of particles with contacts)
            particles_with_contacts = np.sum(degrees > 0)
            net_degree[t] = np.sum(degrees) / particles_with_contacts if particles_with_contacts > 0 else 0
        
        # Calculate P(Z_i)
        max_degree = int(max(all_degrees)) if all_degrees else 0
        degree_counts = np.bincount(all_degrees, minlength=max_degree+1)
        degree_prob = degree_counts / len(all_degrees) if all_degrees else np.zeros(1)
        
        self.network_metrics['degree_distributions'] = degree_distributions
        self.network_metrics['degree_probability'] = degree_prob
        self.network_metrics['max_degree'] = max_degree
        self.network_metrics['mean_degree'] = mean_degree
        self.network_metrics['net_degree'] = net_degree
        self.network_metrics['avg_mean_degree'] = np.mean(mean_degree)
        valid_net_degrees = net_degree[net_degree > 0]
        self.network_metrics['avg_net_degree'] = np.mean(valid_net_degrees) if len(valid_net_degrees) > 0 else 0
        
        return degree_prob, mean_degree, net_degree
    
    def run_analysis(self):
        """Run the frictional contact analysis pipeline"""
        print("Starting frictional contact analysis...")
        print("1. Loading simulation data...")
        self.load_simulation_data()
        
        print("2. Calculating degree distributions...")
        self.calculate_degree_distribution()
        
        print("Analysis complete!")
        return True
    
    def plot_degree_distribution(self, save_fig=True):
        """Plot degree distribution P(Z_i)"""
        if 'degree_probability' not in self.network_metrics:
            self.calculate_degree_distribution()
        
        plt.figure(figsize=(10, 6))
        degree_prob = self.network_metrics['degree_probability']
        degrees = np.arange(len(degree_prob))
        
        plt.bar(degrees, degree_prob, alpha=0.7, color='blue')
        plt.xlabel('Frictional Contact Number $Z_i$')
        plt.ylabel('Probability $P(Z_i)$')
        plt.title(f'Degree Distribution (φ={self.phi}, ar={self.ar})')
        plt.grid(True, alpha=0.3)
        
        if save_fig:
            plt.savefig(f'degree_distribution_phi{self.phi}_ar{self.ar}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def export_results(self, filename=None):
        """Export analysis results to pickle file"""
        if filename is None:
            filename = f'frictional_contact_analysis_phi{self.phi}_ar{self.ar}_vr{self.vr}.pkl'
        
        results = {
            'parameters': {
                'phi': self.phi,
                'ar': self.ar,
                'vr': self.vr,
                'run_num': self.run_num,
                'npp': self.npp
            },
            'bulk_data': self.bulk_data,
            'network_metrics': self.network_metrics
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Results exported to {filename}")
        return filename
    
    def print_summary(self):
        """Print summary of frictional contact analysis"""
        print("="*60)
        print(f"FRICTIONAL CONTACT ANALYSIS SUMMARY")
        print("="*60)
        print(f"System Parameters:")
        print(f"  Volume fraction φ = {self.phi}")
        print(f"  Aspect ratio ar = {self.ar}")
        print(f"  Velocity ratio vr = {self.vr}")
        print(f"  Number of particles = {self.npp}")
        
        print(f"\nFrictional Contact Metrics:")
        print(f"  Average mean degree <Z> = {self.network_metrics['avg_mean_degree']:.2f}")
        print(f"  Average net degree <Z_net> = {self.network_metrics['avg_net_degree']:.2f}")
        print(f"  Maximum degree observed = {self.network_metrics['max_degree']}")
        print("="*60)

def analyze_multiple_systems(data_dir, systems_list, save_results=True):
    """Analyze multiple systems in batch"""
    analyzers = {}
    
    for phi, ar, vr, run_num in systems_list:
        print(f"\n{'='*50}")
        print(f"Analyzing system: φ={phi}, ar={ar}, vr={vr}, run={run_num}")
        print(f"{'='*50}")
        
        try:
            analyzer = FrictionalContactAnalyzer(data_dir, phi, ar, vr, run_num)
            analyzer.run_analysis()
            analyzer.print_summary()
            
            if save_results:
                analyzer.export_results()
            
            key = (phi, ar, vr, run_num)
            analyzers[key] = analyzer
            
        except Exception as e:
            print(f"Error analyzing system φ={phi}, ar={ar}, vr={vr}: {e}")
            continue
    
    return analyzers

def compare_systems(analyzers_dict, save_fig=True):
    """Compare degree distributions and evolution across systems"""
    # Initialize figure for plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(analyzers_dict)))
    
    max_degree = 0
    for (phi, ar, vr, run), analyzer in analyzers_dict.items():
        max_degree = max(max_degree, analyzer.network_metrics['max_degree'])
    
    for i, ((phi, ar, vr, run), analyzer) in enumerate(analyzers_dict.items()):
        color = colors[i]
        label = f'φ={phi}'
        
        # Plot P(Z_i)
        degree_prob = analyzer.network_metrics['degree_probability']
        degrees = np.arange(len(degree_prob))
        ax1.bar(degrees + i*0.2, degree_prob, width=0.2, alpha=0.7, color=color, label=label)
        
        # Plot <Z> and <Z_net>
        shear_rates = analyzer.bulk_data['shear_rate'][:len(analyzer.network_metrics['mean_degree'])]
        ax2.plot(shear_rates, analyzer.network_metrics['mean_degree'], color=color, linestyle='-', linewidth=2, label=f'{label} $\\langle Z \\rangle$')
        ax3.plot(shear_rates, analyzer.network_metrics['net_degree'], color=color, linestyle='--', linewidth=2, label=f'{label} $\\langle Z_{net} \\rangle$')
    
    ax1.set_xlabel('Frictional Contact Number $Z_i$')
    ax1.set_ylabel('Probability $P(Z_i)$')
    ax1.set_title('Degree Distribution Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Shear Rate $\\dot{\\gamma}$')
    ax2.set_ylabel('Mean Degree $\\langle Z \\rangle$')
    ax2.set_title('Mean Degree Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3.set_xlabel('Shear Rate $\\dot{\\gamma}$')
    ax3.set_ylabel('Net Degree $\\langle Z_{net} \\rangle$')
    ax3.set_title('Net Degree Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig('frictional_contact_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Parent directory for all systems
    data_directory = '/Volumes/T7 Shield/NEW KN VALUES FOR SUSPENSION DYNAMICS/Stress 10r'
    
    # Systems to analyze
    systems_to_analyze = [
        (0.54, 1.4, 0.5, 1),
        (0.55, 1.4, 0.5, 1),
        (0.56, 1.4, 0.5, 1),
        (0.57, 1.4, 0.5, 1),
        (0.58, 1.4, 0.5, 1)
    ]
    
    # Run batch analysis
    analyzers = analyze_multiple_systems(data_directory, systems_to_analyze)
    
    # Compare systems
    compare_systems(analyzers)