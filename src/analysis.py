"""
Analysis Module

Comprehensive analysis and visualization of simulation results:
- Scenario comparisons
- District-level analysis
- Statistical tests
- CO2 impact assessment
- Visualization generation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Dict
import logging
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class ResultsAnalyzer:
    """
    Comprehensive analyzer for simulation results.
    """
    
    def __init__(self, results_df: pd.DataFrame):
        """
        Initialize analyzer with simulation results.
        
        Args:
            results_df: DataFrame with simulation results
        """
        self.results = results_df.copy()
        self.summary_stats: Optional[pd.DataFrame] = None
        
    def calculate_summary_statistics(self) -> pd.DataFrame:
        """
        Calculate summary statistics grouped by scenario and district.
        
        Returns:
            DataFrame with mean, std, ci for key metrics
        """
        logger.info("Calculating summary statistics...")
        
        # Group by scenario and district
        grouped = self.results.groupby(['scenario', 'district'])
        
        # Aggregate metrics
        summary = grouped.agg({
            'total_items': ['mean', 'std'],
            'items_reused': ['mean', 'std'],
            'items_collected': ['mean', 'std'],
            'reuse_rate': ['mean', 'std'],
            'avg_time_to_collection': ['mean', 'std'],
            'avg_queue_length': ['mean', 'std'],
            'total_co2_kg': ['mean', 'std']
        })
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary = summary.reset_index()
        
        # Calculate confidence intervals (95%)
        confidence_level = 0.95
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        for metric in ['total_items', 'items_reused', 'reuse_rate', 'total_co2_kg']:
            mean_col = f'{metric}_mean'
            std_col = f'{metric}_std'
            
            if mean_col in summary.columns and std_col in summary.columns:
                n = len(self.results.groupby(['scenario', 'district']).size())
                summary[f'{metric}_ci'] = z_score * summary[std_col] / np.sqrt(n)
        
        self.summary_stats = summary
        logger.info("Summary statistics calculated")
        
        return summary
    
    def compare_scenarios(self, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Compare all scenarios against baseline.
        
        Args:
            output_path: Optional path to save comparison table
            
        Returns:
            DataFrame with scenario comparisons
        """
        logger.info("Comparing scenarios...")
        
        if self.summary_stats is None:
            self.calculate_summary_statistics()
        
        # Get baseline metrics
        baseline = self.summary_stats[self.summary_stats['scenario'] == 'baseline']
        baseline_reuse = baseline['reuse_rate_mean'].mean()
        baseline_co2 = baseline['total_co2_kg_mean'].sum()
        
        # Calculate improvements
        comparison = []
        
        for scenario in ['pessimistic', 'realistic', 'optimistic']:
            scenario_data = self.summary_stats[self.summary_stats['scenario'] == scenario]
            
            reuse_rate = scenario_data['reuse_rate_mean'].mean()
            co2_total = scenario_data['total_co2_kg_mean'].sum()
            
            reuse_improvement = ((reuse_rate - baseline_reuse) / baseline_reuse) * 100
            co2_reduction = ((baseline_co2 - co2_total) / baseline_co2) * 100
            
            comparison.append({
                'scenario': scenario.capitalize(),
                'avg_reuse_rate': f"{reuse_rate:.1%}",
                'reuse_improvement_vs_baseline': f"{reuse_improvement:+.1f}%",
                'total_co2_kg': f"{co2_total:.0f}",
                'co2_reduction_vs_baseline': f"{co2_reduction:+.1f}%"
            })
        
        comparison_df = pd.DataFrame(comparison)
        
        if output_path:
            comparison_df.to_csv(output_path, index=False)
            logger.info(f"Comparison saved to {output_path}")
        
        return comparison_df
    
    def analyze_by_district(self, top_n: int = 10) -> pd.DataFrame:
        """
        Identify districts with highest impact potential.
        
        Args:
            top_n: Number of top districts to return
            
        Returns:
            DataFrame with top districts ranked by impact
        """
        logger.info("Analyzing by district...")
        
        # Calculate improvement for realistic scenario vs baseline
        baseline = self.results[self.results['scenario'] == 'baseline']
        realistic = self.results[self.results['scenario'] == 'realistic']
        
        baseline_by_district = baseline.groupby('district').agg({
            'reuse_rate': 'mean',
            'total_co2_kg': 'mean'
        }).rename(columns={'reuse_rate': 'baseline_reuse', 'total_co2_kg': 'baseline_co2'})
        
        realistic_by_district = realistic.groupby('district').agg({
            'reuse_rate': 'mean',
            'total_co2_kg': 'mean'
        }).rename(columns={'reuse_rate': 'realistic_reuse', 'total_co2_kg': 'realistic_co2'})
        
        # Merge
        district_comparison = baseline_by_district.join(realistic_by_district)
        
        # Calculate improvements
        district_comparison['reuse_improvement'] = (
            (district_comparison['realistic_reuse'] - district_comparison['baseline_reuse']) / 
            district_comparison['baseline_reuse']
        ) * 100
        
        district_comparison['co2_reduction'] = (
            (district_comparison['baseline_co2'] - district_comparison['realistic_co2']) / 
            district_comparison['baseline_co2']
        ) * 100
        
        # Sort by improvement
        district_comparison = district_comparison.sort_values('reuse_improvement', ascending=False)
        
        logger.info(f"Top {top_n} districts identified")
        
        return district_comparison.head(top_n)
    
    def plot_scenario_comparison(self, output_dir: Optional[str] = None):
        """
        Create scenario comparison visualizations.
        
        Args:
            output_dir: Directory to save plots
        """
        logger.info("Generating scenario comparison plots...")
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Plot 1: Reuse Rate by Scenario
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scenario_order = ['baseline', 'pessimistic', 'realistic', 'optimistic']
        scenario_data = self.results.groupby('scenario')['reuse_rate'].mean().reindex(scenario_order)
        
        colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
        bars = ax.bar(range(len(scenario_data)), scenario_data.values, color=colors, alpha=0.8)
        
        ax.set_xticks(range(len(scenario_data)))
        ax.set_xticklabels([s.capitalize() for s in scenario_data.index], rotation=0)
        ax.set_ylabel('Reuse Rate (%)')
        ax.set_title('Average Reuse Rate by Scenario', fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(scenario_data.values) * 1.2)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1%}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(output_path / 'scenario_reuse_comparison.png', dpi=300, bbox_inches='tight')
            logger.info("Saved: scenario_reuse_comparison.png")
        
        plt.show()
        
        # Plot 2: CO2 Emissions by Scenario
        fig, ax = plt.subplots(figsize=(10, 6))
        
        co2_data = self.results.groupby('scenario')['total_co2_kg'].sum().reindex(scenario_order)
        
        bars = ax.bar(range(len(co2_data)), co2_data.values, color=colors, alpha=0.8)
        
        ax.set_xticks(range(len(co2_data)))
        ax.set_xticklabels([s.capitalize() for s in co2_data.index], rotation=0)
        ax.set_ylabel('Total CO₂ Emissions (kg)')
        ax.set_title('Total CO₂ Emissions by Scenario', fontsize=14, fontweight='bold')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:,.0f} kg',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(output_path / 'scenario_co2_comparison.png', dpi=300, bbox_inches='tight')
            logger.info("Saved: scenario_co2_comparison.png")
        
        plt.show()
        
        # Plot 3: Box plot of reuse rates
        fig, ax = plt.subplots(figsize=(10, 6))
        
        self.results['scenario_label'] = pd.Categorical(
            self.results['scenario'],
            categories=scenario_order,
            ordered=True
        )
        
        sns.boxplot(data=self.results, x='scenario_label', y='reuse_rate', 
                    palette=colors, ax=ax)
        
        ax.set_xlabel('Scenario')
        ax.set_ylabel('Reuse Rate')
        ax.set_title('Distribution of Reuse Rates by Scenario', fontsize=14, fontweight='bold')
        ax.set_xticklabels([s.capitalize() for s in scenario_order])
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(output_path / 'scenario_reuse_distribution.png', dpi=300, bbox_inches='tight')
            logger.info("Saved: scenario_reuse_distribution.png")
        
        plt.show()
    
    def plot_district_analysis(self, top_n: int = 10, output_dir: Optional[str] = None):
        """
        Create district-level analysis plots.
        
        Args:
            top_n: Number of top districts to visualize
            output_dir: Directory to save plots
        """
        logger.info("Generating district analysis plots...")
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        district_data = self.analyze_by_district(top_n=top_n)
        
        # Plot: Top districts by improvement
        fig, ax = plt.subplots(figsize=(12, 8))
        
        y_pos = np.arange(len(district_data))
        
        ax.barh(y_pos, district_data['reuse_improvement'].values, color='#2ca02c', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(district_data.index)
        ax.invert_yaxis()
        ax.set_xlabel('Reuse Rate Improvement vs Baseline (%)')
        ax.set_title(f'Top {top_n} Districts by Reuse Improvement (Realistic Scenario)', 
                     fontsize=14, fontweight='bold')
        
        # Add value labels
        for i, v in enumerate(district_data['reuse_improvement'].values):
            ax.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(output_path / 'top_districts_improvement.png', dpi=300, bbox_inches='tight')
            logger.info("Saved: top_districts_improvement.png")
        
        plt.show()
    
    def generate_full_report(self, output_dir: str = "outputs/figures"):
        """
        Generate complete analysis report with all visualizations.
        
        Args:
            output_dir: Directory to save all outputs
        """
        logger.info("Generating full analysis report...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Calculate summary statistics
        summary = self.calculate_summary_statistics()
        summary.to_csv(output_path / 'summary_statistics.csv', index=False)
        logger.info("Saved: summary_statistics.csv")
        
        # Scenario comparison
        comparison = self.compare_scenarios(output_path / 'scenario_comparison.csv')
        logger.info("Saved: scenario_comparison.csv")
        
        # District analysis
        district_analysis = self.analyze_by_district(top_n=15)
        district_analysis.to_csv(output_path / 'district_analysis.csv')
        logger.info("Saved: district_analysis.csv")
        
        # Generate all plots
        self.plot_scenario_comparison(output_dir=output_dir)
        self.plot_district_analysis(top_n=10, output_dir=output_dir)
        
        logger.info(f"Full report generated in {output_dir}")
        
        # Print summary to console
        print("\n" + "="*60)
        print("SIMULATION RESULTS SUMMARY")
        print("="*60)
        print("\nScenario Comparison:")
        print(comparison.to_string(index=False))
        print("\n" + "="*60)


if __name__ == "__main__":
    # Test analysis
    # Load sample results
    results = pd.read_csv("outputs/logs/simulation_results.csv")
    
    analyzer = ResultsAnalyzer(results)
    analyzer.generate_full_report()