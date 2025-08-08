"""
AI Development Workflow Diagram Generator

Comprehensive workflow diagrams for AI development,
including both high-level overview and detailed healthcare-specific implementation.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

class AIWorkflowDiagramGenerator:
    """
    Generate comprehensive AI development workflow diagrams
    """
    
    def __init__(self):
        self.colors = {
            'problem': '#E3F2FD',      # Light blue
            'data': '#F3E5F5',         # Light purple
            'model': '#E8F5E8',        # Light green
            'evaluation': '#FFF3E0',   # Light orange
            'deployment': '#FFEBEE',   # Light red
            'monitoring': '#F1F8E9',   # Light lime
            'decision': '#FFFDE7',     # Light yellow
            'iteration': '#FCE4EC'     # Light pink
        }
        
    def create_main_workflow_diagram(self):
        """
        Create the main AI development workflow diagram
        """
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 14)
        ax.axis('off')
        
        # Title
        ax.text(5, 13.5, 'AI Development Workflow for Healthcare Applications', 
                fontsize=20, fontweight='bold', ha='center')
        
        # Define workflow stages with positions and connections
        stages = [
            # (x, y, width, height, text, color_key, stage_type)
            (1, 12, 2, 0.8, '1. Problem Definition\n• Define objectives\n• Identify stakeholders\n• Set success metrics', 'problem', 'process'),
            (4, 12, 2, 0.8, '2. Data Understanding\n• Assess data sources\n• Identify limitations\n• Privacy considerations', 'data', 'process'),
            (7, 12, 2, 0.8, '3. Data Collection\n• Gather datasets\n• Data integration\n• Quality assessment', 'data', 'process'),
            
            (1, 10.5, 2, 0.8, '4. Data Preprocessing\n• Handle missing values\n• Feature engineering\n• Data transformation', 'data', 'process'),
            (4, 10.5, 2, 0.8, '5. Exploratory Analysis\n• Statistical analysis\n• Visualization\n• Pattern discovery', 'data', 'process'),
            (7, 10.5, 2, 0.8, '6. Feature Selection\n• Relevance analysis\n• Dimensionality reduction\n• Domain expertise', 'data', 'process'),
            
            (1, 9, 2, 0.8, '7. Model Selection\n• Algorithm choice\n• Architecture design\n• Baseline models', 'model', 'process'),
            (4, 9, 2, 0.8, '8. Model Training\n• Parameter learning\n• Cross-validation\n• Hyperparameter tuning', 'model', 'process'),
            (7, 9, 2, 0.8, '9. Model Evaluation\n• Performance metrics\n• Validation testing\n• Bias assessment', 'evaluation', 'process'),
            
            (4, 7.5, 2, 0.8, 'Performance\nAcceptable?', 'decision', 'decision'),
            
            (1, 6, 2, 0.8, '10. Model Validation\n• Independent testing\n• Clinical validation\n• Regulatory review', 'evaluation', 'process'),
            (4, 6, 2, 0.8, '11. Deployment Planning\n• Integration strategy\n• Infrastructure setup\n• User training', 'deployment', 'process'),
            (7, 6, 2, 0.8, '12. System Integration\n• API development\n• EHR integration\n• Security implementation', 'deployment', 'process'),
            
            (1, 4.5, 2, 0.8, '13. User Testing\n• Acceptance testing\n• Workflow validation\n• Feedback collection', 'deployment', 'process'),
            (4, 4.5, 2, 0.8, '14. Production Deploy\n• Go-live process\n• Monitoring setup\n• Documentation', 'deployment', 'process'),
            (7, 4.5, 2, 0.8, '15. Performance Monitor\n• Accuracy tracking\n• Concept drift detection\n• User feedback', 'monitoring', 'process'),
            
            (4, 3, 2, 0.8, 'Concept Drift\nDetected?', 'decision', 'decision'),
            
            (1, 1.5, 2, 0.8, '16. Model Updates\n• Retrain with new data\n• Algorithm improvements\n• Feature updates', 'iteration', 'process'),
            (4, 1.5, 2, 0.8, '17. Continuous Learning\n• Online learning\n• Feedback integration\n• Performance optimization', 'iteration', 'process'),
            (7, 1.5, 2, 0.8, '18. Knowledge Transfer\n• Best practices\n• Lessons learned\n• Documentation updates', 'iteration', 'process'),
        ]
        
        # Draw boxes
        boxes = {}
        for i, (x, y, w, h, text, color_key, stage_type) in enumerate(stages):
            if stage_type == 'decision':
                # Diamond shape for decisions
                diamond = mpatches.RegularPolygon((x+w/2, y+h/2), 4, radius=0.6,
                                                orientation=np.pi/4, 
                                                facecolor=self.colors[color_key],
                                                edgecolor='black', linewidth=2)
                ax.add_patch(diamond)
            else:
                # Rectangle for processes
                box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                   facecolor=self.colors[color_key],
                                   edgecolor='black', linewidth=2)
                ax.add_patch(box)
            
            # Add text
            ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
                   fontsize=9, fontweight='bold', wrap=True)
            
            boxes[i] = (x + w/2, y + h/2)
        
        # Draw connections
        connections = [
            # Main flow
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
            (5, 6), (6, 7), (7, 8), (8, 9),
            # Decision branch
            (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15),
            # Another decision
            (15, 16), (16, 17), (17, 18),
            # Feedback loops
            (9, 6),  # Performance not acceptable -> back to model selection
            (16, 6), # Model updates -> back to model selection
        ]
        
        for start, end in connections:
            start_pos = boxes[start]
            end_pos = boxes[end]
            
            if start == 9 and end == 6:  # No feedback loop
                ax.annotate('', xy=end_pos, xytext=start_pos,
                           arrowprops=dict(arrowstyle='->', lw=2, color='red',
                                         connectionstyle="arc3,rad=-0.3"))
                ax.text(0.5, 8.2, 'No', fontsize=10, fontweight='bold', color='red')
            elif start == 16 and end == 6:  # Drift detected feedback
                ax.annotate('', xy=end_pos, xytext=start_pos,
                           arrowprops=dict(arrowstyle='->', lw=2, color='orange',
                                         connectionstyle="arc3,rad=0.3"))
                ax.text(8.5, 2.7, 'Yes', fontsize=10, fontweight='bold', color='orange')
            else:
                ax.annotate('', xy=end_pos, xytext=start_pos,
                           arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
        
        # Add Yes/No labels for decisions
        ax.text(5.5, 7.2, 'Yes', fontsize=10, fontweight='bold', color='green')
        ax.text(2.5, 2.7, 'No', fontsize=10, fontweight='bold', color='green')
        
        # Add legend
        legend_elements = [
            mpatches.Rectangle((0, 0), 1, 1, facecolor=self.colors['problem'], 
                             edgecolor='black', label='Problem Definition'),
            mpatches.Rectangle((0, 0), 1, 1, facecolor=self.colors['data'], 
                             edgecolor='black', label='Data Management'),
            mpatches.Rectangle((0, 0), 1, 1, facecolor=self.colors['model'], 
                             edgecolor='black', label='Model Development'),
            mpatches.Rectangle((0, 0), 1, 1, facecolor=self.colors['evaluation'], 
                             edgecolor='black', label='Evaluation & Validation'),
            mpatches.Rectangle((0, 0), 1, 1, facecolor=self.colors['deployment'], 
                             edgecolor='black', label='Deployment'),
            mpatches.Rectangle((0, 0), 1, 1, facecolor=self.colors['monitoring'], 
                             edgecolor='black', label='Monitoring'),
            mpatches.Rectangle((0, 0), 1, 1, facecolor=self.colors['iteration'], 
                             edgecolor='black', label='Iteration & Learning')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        plt.tight_layout()
        plt.show()
        return fig
    
    def create_healthcare_specific_workflow(self):
        """
        Create healthcare-specific AI workflow diagram
        """
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis('off')
        
        # Title
        ax.text(5, 11.5, 'Healthcare AI Workflow: Patient Readmission Prediction', 
                fontsize=18, fontweight='bold', ha='center')
        
        # Healthcare-specific stages
        healthcare_stages = [
            (1, 10, 2.5, 1, 'Clinical Problem\nIdentification\n• 30-day readmission risk\n• Stakeholder alignment\n• Regulatory requirements', 'problem'),
            (6, 10, 2.5, 1, 'Data Governance\n& Privacy\n• HIPAA compliance\n• IRB approval\n• Data use agreements', 'data'),
            
            (1, 8.5, 2.5, 1, 'EHR Data Integration\n• Patient demographics\n• Clinical history\n• Lab results\n• Medications', 'data'),
            (6, 8.5, 2.5, 1, 'Feature Engineering\n• Comorbidity indices\n• Risk stratification\n• Temporal patterns\n• Social determinants', 'data'),
            
            (1, 7, 2.5, 1, 'Model Development\n• Algorithm selection\n• Interpretability focus\n• Bias assessment\n• Clinical validation', 'model'),
            (6, 7, 2.5, 1, 'Performance Evaluation\n• Clinical metrics\n• Fairness testing\n• Subgroup analysis\n• External validation', 'evaluation'),
            
            (1, 5.5, 2.5, 1, 'Clinical Integration\n• EHR embedding\n• Workflow design\n• Alert systems\n• Decision support', 'deployment'),
            (6, 5.5, 2.5, 1, 'User Training\n• Clinician education\n• System familiarization\n• Best practices\n• Feedback mechanisms', 'deployment'),
            
            (1, 4, 2.5, 1, 'Go-Live & Monitoring\n• Pilot deployment\n• Performance tracking\n• User satisfaction\n• Clinical outcomes', 'monitoring'),
            (6, 4, 2.5, 1, 'Continuous Improvement\n• Model updates\n• Feature refinement\n• Process optimization\n• Knowledge sharing', 'iteration'),
        ]
        
        # Draw healthcare workflow
        for i, (x, y, w, h, text, color_key) in enumerate(healthcare_stages):
            box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                               facecolor=self.colors[color_key],
                               edgecolor='black', linewidth=2)
            ax.add_patch(box)
            
            ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
                   fontsize=10, fontweight='bold')
        
        # Add arrows between stages
        arrow_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)]
        
        for start_idx, end_idx in arrow_pairs:
            start_stage = healthcare_stages[start_idx]
            end_stage = healthcare_stages[end_idx]
            
            start_x = start_stage[0] + start_stage[2]/2
            start_y = start_stage[1]
            end_x = end_stage[0] + end_stage[2]/2
            end_y = end_stage[1] + end_stage[3]
            
            if start_idx % 2 == end_idx % 2:  # Same column
                ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                           arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
            else:  # Different columns
                ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                           arrowprops=dict(arrowstyle='->', lw=2, color='blue',
                                         connectionstyle="arc3,rad=0.3"))
        
        # Add feedback loop
        ax.annotate('', xy=(2.25, 7.5), xytext=(7.25, 4.5),
                   arrowprops=dict(arrowstyle='->', lw=2, color='red',
                                 connectionstyle="arc3,rad=0.5"))
        ax.text(4.5, 2.5, 'Continuous Feedback Loop', ha='center', 
               fontsize=12, fontweight='bold', color='red')
        
        # Add regulatory compliance box
        compliance_box = FancyBboxPatch((0.5, 2), 9, 1, boxstyle="round,pad=0.1",
                                      facecolor='#FFCDD2', edgecolor='red', 
                                      linewidth=2, linestyle='--')
        ax.add_patch(compliance_box)
        ax.text(5, 2.5, 'Regulatory Compliance Framework\n'
                       'FDA Guidelines • HIPAA • Clinical Evidence Standards • Quality Assurance',
                ha='center', va='center', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        return fig
    
    def create_crisp_dm_diagram(self):
        """
        Create CRISP-DM methodology diagram
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.axis('off')
        
        # Title
        ax.text(0, 5.5, 'CRISP-DM Methodology for Healthcare AI', 
                fontsize=18, fontweight='bold', ha='center')
        
        # CRISP-DM phases in circular arrangement
        phases = [
            (0, 4, 'Business\nUnderstanding', 'problem'),
            (3.5, 2, 'Data\nUnderstanding', 'data'),
            (3.5, -2, 'Data\nPreparation', 'data'),
            (0, -4, 'Modeling', 'model'),
            (-3.5, -2, 'Evaluation', 'evaluation'),
            (-3.5, 2, 'Deployment', 'deployment')
        ]
        
        # Draw phases as circles
        for x, y, text, color_key in phases:
            circle = plt.Circle((x, y), 1.2, facecolor=self.colors[color_key],
                              edgecolor='black', linewidth=3)
            ax.add_patch(circle)
            ax.text(x, y, text, ha='center', va='center', 
                   fontsize=11, fontweight='bold')
        
        # Draw arrows between phases
        connections = [
            ((0, 4), (3.5, 2)),      # Business -> Data Understanding
            ((3.5, 2), (3.5, -2)),   # Data Understanding -> Data Preparation
            ((3.5, -2), (0, -4)),    # Data Preparation -> Modeling
            ((0, -4), (-3.5, -2)),   # Modeling -> Evaluation
            ((-3.5, -2), (-3.5, 2)), # Evaluation -> Deployment
            ((-3.5, 2), (0, 4))      # Deployment -> Business Understanding
        ]
        
        for (start_x, start_y), (end_x, end_y) in connections:
            # Calculate arrow start and end points on circle edges
            angle_start = np.arctan2(end_y - start_y, end_x - start_x)
            angle_end = np.arctan2(start_y - end_y, start_x - end_x)
            
            start_edge_x = start_x + 1.2 * np.cos(angle_start)
            start_edge_y = start_y + 1.2 * np.sin(angle_start)
            end_edge_x = end_x + 1.2 * np.cos(angle_end)
            end_edge_y = end_y + 1.2 * np.sin(angle_end)
            
            ax.annotate('', xy=(end_edge_x, end_edge_y), xytext=(start_edge_x, start_edge_y),
                       arrowprops=dict(arrowstyle='->', lw=3, color='blue'))
        
        # Add inner connections (iterative nature)
        inner_connections = [
            ((3.5, 2), (3.5, -2), (0, -4)),    # Data Understanding <-> Preparation <-> Modeling
            ((0, -4), (-3.5, -2))              # Modeling <-> Evaluation
        ]
        
        # Add data at center
        data_circle = plt.Circle((0, 0), 0.8, facecolor='#FFF59D',
                               edgecolor='black', linewidth=2)
        ax.add_patch(data_circle)
        ax.text(0, 0, 'Data', ha='center', va='center', 
               fontsize=14, fontweight='bold')
        
        # Add phase descriptions
        descriptions = [
            (0, 6, 'Define business objectives\nand success criteria'),
            (5.5, 2, 'Collect and explore\ninitial dataset'),
            (5.5, -2, 'Clean and transform\ndata for modeling'),
            (0, -6, 'Select and build\npredictive models'),
            (-5.5, -2, 'Assess model quality\nand business value'),
            (-5.5, 2, 'Deploy model into\nproduction environment')
        ]
        
        for x, y, desc in descriptions:
            ax.text(x, y, desc, ha='center', va='center', 
                   fontsize=9, style='italic',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        return fig
    
    def create_all_diagrams(self):
        """
        Create all workflow diagrams
        """
        print("Generating AI Development Workflow Diagrams...")
        print("=" * 50)
        
        print("1. Creating Main AI Development Workflow...")
        main_fig = self.create_main_workflow_diagram()
        
        print("2. Creating Healthcare-Specific Workflow...")
        healthcare_fig = self.create_healthcare_specific_workflow()
        
        print("3. Creating CRISP-DM Methodology Diagram...")
        crisp_dm_fig = self.create_crisp_dm_diagram()
        
        print("\nAll diagrams generated successfully!")
        
        return main_fig, healthcare_fig, crisp_dm_fig

# Example usage
if __name__ == "__main__":
    # Initialize diagram generator
    diagram_generator = AIWorkflowDiagramGenerator()
    
    # Create all diagrams
    main_fig, healthcare_fig, crisp_dm_fig = diagram_generator.create_all_diagrams()
    
    print("\nDiagram Features:")
    print("• Main Workflow: Complete 18-stage AI development process")
    print("• Healthcare Workflow: Domain-specific implementation steps")
    print("• CRISP-DM: Industry-standard methodology visualization")
    print("• Color-coded stages for easy identification")
    print("• Feedback loops and iterative processes highlighted")
