import os
import numpy as np
import pandas as pd
from scapy.all import rdpcap
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
import torch
import glob
from tqdm import tqdm
from collections import defaultdict
import json

class MAGNNPreprocessor:
    def __init__(self, output_dir='processed/', temporal_window=300):
        self.output_dir = output_dir
        self.temporal_window = temporal_window  # seconds
        os.makedirs(output_dir, exist_ok=True)
        
        # Feature configuration
        self.node_features = [
            'total_packets', 'total_bytes', 'avg_pkt_size',
            'duration', 'packet_rate', 'byte_rate'
        ]
        self.edge_features = [
            'protocol', 'packet_count', 'total_bytes',
            'duration', 'start_time', 'end_time'
        ]
        
    def process_all_datasets(self):
        """Process all supported datasets"""
        print("Starting MAGNN data preprocessing...")
        self.process_ctu13()
        self.process_iscxvpn2016()
        self.process_cicids2017()
        self.process_dohbrw2020()
        print("All datasets processed successfully!")
        
    def process_ctu13(self, input_dir='data/CTU-13/'):
        """Process CTU-13 botnet dataset"""
        print("\nProcessing CTU-13 dataset...")
        scenarios = glob.glob(os.path.join(input_dir, '*/'))
        
        for scenario in tqdm(scenarios, desc="CTU-13 Scenarios"):
            # Read label files
            with open(os.path.join(scenario, 'botnet_label.txt')) as f:
                malicious_ips = set(line.strip() for line in f)
            
            # Process PCAP files
            pcap_files = glob.glob(os.path.join(scenario, '*.pcap'))
            graphs = []
            
            for pcap in pcap_files:
                # Extract flows and build graph
                graph = self._pcap_to_graph(pcap, malicious_ips)
                if graph:
                    graphs.append(graph)
            
            # Save processed data
            scenario_name = os.path.basename(os.path.normpath(scenario))
            self._save_dataset(graphs, f'ctu13_{scenario_name}')
    
    def process_iscxvpn2016(self, input_dir='data/ISCXVPN2016/'):
        """Process ISCXVPN2016 dataset"""
        print("\nProcessing ISCXVPN2016 dataset...")
        app_dirs = glob.glob(os.path.join(input_dir, '*/'))
        
        graphs = []
        for app_dir in tqdm(app_dirs, desc="Applications"):
            # VPN vs non-VPN classification
            is_vpn = 'VPN' in os.path.basename(app_dir)
            
            # Process all PCAPs in directory
            pcap_files = glob.glob(os.path.join(app_dir, '*.pcap'))
            for pcap in pcap_files:
                graph = self._pcap_to_graph(pcap, malicious_ips=None, label=is_vpn)
                if graph:
                    graphs.append(graph)
        
        self._save_dataset(graphs, 'iscxvpn2016')
    
    def process_cicids2017(self, input_dir='data/CICIDS2017/'):
        """Process CICIDS2017 dataset"""
        print("\nProcessing CICIDS2017 dataset...")
        csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
        
        # Load and concatenate all CSV files
        dfs = []
        for csv in tqdm(csv_files, desc="Loading CSVs"):
            df = pd.read_csv(csv)
            dfs.append(df)
        full_df = pd.concat(dfs)
        
        # Preprocess and create graphs
        graphs = self._csv_to_graphs(full_df)
        self._save_dataset(graphs, 'cicids2017')
    
    def process_dohbrw2020(self, input_dir='data/DoHBrw2020/'):
        """Process DNS-over-HTTPS dataset"""
        print("\nProcessing DoHBrw2020 dataset...")
        pcap_files = glob.glob(os.path.join(input_dir, '*.pcap'))
        
        graphs = []
        for pcap in tqdm(pcap_files, desc="PCAP Files"):
            # DoH traffic is considered malicious in this context
            is_doh = 'malicious' in os.path.basename(pcap).lower()
            graph = self._pcap_to_graph(pcap, malicious_ips=None, label=is_doh)
            if graph:
                graphs.append(graph)
        
        self._save_dataset(graphs, 'dohbrw2020')
    
    def _pcap_to_graph(self, pcap_file, malicious_ips=None, label=None):
        """Convert PCAP file to graph representation"""
        try:
            packets = rdpcap(pcap_file)
        except:
            return None
        
        # Extract flows (conversations)
        flows = defaultdict(list)
        for pkt in packets:
            if 'IP' in pkt:
                src = pkt['IP'].src
                dst = pkt['IP'].dst
                flow_key = tuple(sorted((src, dst)))
                flows[flow_key].append(pkt)
        
        # Build node and edge features
        node_features = {}
        edge_data = []
        
        for (src, dst), pkts in flows.items():
            # Edge features
            start_time = pkts[0].time
            end_time = pkts[-1].time
            duration = end_time - start_time
            total_bytes = sum(len(p) for p in pkts)
            
            edge_data.append({
                'src': src,
                'dst': dst,
                'protocol': self._get_protocol(pkts[0]),
                'packet_count': len(pkts),
                'total_bytes': total_bytes,
                'duration': duration,
                'start_time': start_time,
                'end_time': end_time
            })
            
            # Update node features (both src and dst)
            for ip in [src, dst]:
                if ip not in node_features:
                    node_features[ip] = {
                        'total_packets': 0,
                        'total_bytes': 0,
                        'packet_times': []
                    }
                node_features[ip]['total_packets'] += len(pkts)
                node_features[ip]['total_bytes'] += total_bytes
                node_features[ip]['packet_times'].extend([p.time for p in pkts])
        
        # Finalize node features
        nodes = []
        features = []
        labels = []
        
        for ip, stats in node_features.items():
            # Calculate temporal features
            packet_times = sorted(stats['packet_times'])
            time_diffs = np.diff(packet_times)
            
            node_feature = [
                stats['total_packets'],
                stats['total_bytes'],
                stats['total_bytes'] / stats['total_packets'] if stats['total_packets'] > 0 else 0,
                packet_times[-1] - packet_times[0] if len(packet_times) > 1 else 0,
                len(packet_times) / self.temporal_window,
                stats['total_bytes'] / self.temporal_window
            ]
            
            nodes.append(ip)
            features.append(node_feature)
            
            # Node labels (if malicious IPs provided)
            if malicious_ips is not None:
                labels.append(1 if ip in malicious_ips else 0)
            elif label is not None:
                labels.append(label)
        
        # Create edge index and edge attributes
        node_map = {ip: i for i, ip in enumerate(nodes)}
        edge_index = []
        edge_attr = []
        
        for edge in edge_data:
            src_idx = node_map[edge['src']]
            dst_idx = node_map[edge['dst']]
            edge_index.append([src_idx, dst_idx])
            
            # Normalize edge features
            edge_attr.append([
                edge['protocol'],
                edge['packet_count'],
                edge['total_bytes'],
                edge['duration'],
                edge['start_time'] % self.temporal_window,  # Relative time
                edge['end_time'] % self.temporal_window
            ])
        
        # Convert to tensors
        x = torch.FloatTensor(features)
        edge_index = torch.LongTensor(edge_index).t().contiguous()
        edge_attr = torch.FloatTensor(edge_attr)
        y = torch.LongTensor(labels) if labels else None
        
        # Temporal features (sliding window aggregation)
        temporal_feats = self._create_temporal_features(packets)
        
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            temporal=torch.FloatTensor(temporal_feats)
        )
    
    def _csv_to_graphs(self, df):
        """Convert CICIDS2017 CSV data to graphs"""
        # Group by time windows
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['time_window'] = (df['Timestamp'].astype(int) // 
                           (self.temporal_window * 1e9)).astype(int)
        
        graphs = []
        for window, group in tqdm(df.groupby('time_window'), desc="Time Windows"):
            # Create graph for this time window
            nodes = set()
            edge_data = []
            
            # Process each flow in the window
            for _, row in group.iterrows():
                src = row['Source IP']
                dst = row['Destination IP']
                nodes.update([src, dst])
                
                edge_data.append({
                    'src': src,
                    'dst': dst,
                    'protocol': self._protocol_to_num(row['Protocol']),
                    'packet_count': row['Total Fwd Packets'] + row['Total Bwd Packets'],
                    'total_bytes': row['Total Length of Fwd Packets'] + row['Total Length of Bwd Packets'],
                    'duration': row['Flow Duration'],
                    'label': 1 if row['Label'] != 'BENIGN' else 0
                })
            
            # Build graph (similar to _pcap_to_graph)
            graph = self._build_graph_from_edges(edge_data)
            if graph:
                graphs.append(graph)
        
        return graphs
    
    def _build_graph_from_edges(self, edge_data):
        """Helper to build graph from edge data"""
        # Similar to _pcap_to_graph but for CSV data
        pass  # Implementation similar to _pcap_to_graph
    
    def _create_temporal_features(self, packets):
        """Create temporal features from packet timestamps"""
        if not packets:
            return np.zeros(5)
        
        times = [p.time for p in packets]
        time_diffs = np.diff(sorted(times))
        
        return [
            len(packets) / self.temporal_window,  # Packet rate
            sum(len(p) for p in packets) / self.temporal_window,  # Byte rate
            np.mean(time_diffs) if len(time_diffs) > 0 else 0,  # Avg inter-arrival
            np.std(time_diffs) if len(time_diffs) > 0 else 0,   # Std inter-arrival
            times[-1] - times[0] if len(times) > 1 else 0       # Duration
        ]
    
    def _get_protocol(self, packet):
        """Extract protocol numeric code"""
        proto_map = {'TCP': 0, 'UDP': 1, 'ICMP': 2}
        for layer in packet.layers():
            if layer.name in proto_map:
                return proto_map[layer.name]
        return 3  # Unknown
    
    def _protocol_to_num(self, protocol_str):
        """Convert protocol string to numeric code"""
        protocol_str = str(protocol_str).upper()
        if 'TCP' in protocol_str:
            return 0
        elif 'UDP' in protocol_str:
            return 1
        elif 'ICMP' in protocol_str:
            return 2
        return 3
    
    def _save_dataset(self, graphs, dataset_name):
        """Save processed dataset with train/val/test split"""
        if not graphs:
            return
        
        # Split datasets (70/15/15)
        np.random.shuffle(graphs)
        n = len(graphs)
        train = graphs[:int(0.7*n)]
        val = graphs[int(0.7*n):int(0.85*n)]
        test = graphs[int(0.85*n):]
        
        # Save to files
        save_dir = os.path.join(self.output_dir, dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save(train, os.path.join(save_dir, 'train.pt'))
        torch.save(val, os.path.join(save_dir, 'val.pt'))
        torch.save(test, os.path.join(save_dir, 'test.pt'))
        
        # Save feature metadata
        meta = {
            'node_features': self.node_features,
            'edge_features': self.edge_features,
            'num_classes': 2,
            'temporal_dim': 5
        }
        with open(os.path.join(save_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f)
        
        print(f"Saved {dataset_name}: {len(train)} train, {len(val)} val, {len(test)} test graphs")

# Usage example
if __name__ == "__main__":
    preprocessor = MAGNNPreprocessor(output_dir='processed/')
    preprocessor.process_all_datasets()
