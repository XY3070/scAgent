import unittest
import pandas as pd
from unittest.mock import MagicMock, patch
import os
import json
import logging

# Configure logging for the test module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory of SRAgent to the Python path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Now you can import SRAgent as a top-level package
from SRAgent.db.categorization_logic import categorize_datasets_by_project, group_datasets_by_project_id
from SRAgent.db.categorize import get_project_statistics, create_classify_ready_export, run_export_workflow

class TestCategorizeModule(unittest.TestCase):

    def setUp(self):
        self.output_json_path = "test_workflow_export.json"
        if os.path.exists(self.output_json_path):
            os.remove(self.output_json_path)

    def tearDown(self):
        if os.path.exists(self.output_json_path):
            os.remove(self.output_json_path)

    def test_categorize_module_full_workflow(self):
        logger.info("\n--- Running test_categorize_module_full_workflow ---")

        # Sample data for testing
        data = {
            'sra_id': ['SRR123', 'SRR456', 'SRR789', 'SRR101', 'SRR102', 'SRR103'],
            'project_id': ['PRJNA1', 'PRJNA2', 'GSE123', 'E-MTAB-123', 'ena-STUDY-456789', 'Other'],
            'organism': ['human', 'mouse', 'human', 'human', 'human', 'human'],
            'title': ['Sample 1', 'Sample 2', 'Sample 3', 'Sample 4', 'Sample 5', 'Sample 6'],
            'study_alias': ['PRJNA1', 'PRJNA2', 'GSE123', 'E-MTAB-123', 'ena-STUDY-456789', 'Other'],
            'gse_title': ['', '', 'GSE123_title', '', '', ''],
            'sra_ID': ['SRR123', 'SRR456', 'SRR789', 'SRR101', 'SRR102', 'SRR103']
        }
        df = pd.DataFrame(data)

        # Test categorize_datasets_by_project
        logger.info("Testing categorize_datasets_by_project...")
        categorized = categorize_datasets_by_project(df)
        self.assertIn('GSE', categorized)
        self.assertEqual(len(categorized['GSE']), 1)
        self.assertIn('PRJNA', categorized)
        self.assertEqual(len(categorized['PRJNA']), 2)
        self.assertIn('ena-STUDY', categorized)
        self.assertEqual(len(categorized['ena-STUDY']), 1)
        self.assertIn('E-MTAB', categorized)
        self.assertEqual(len(categorized['E-MTAB']), 1)
        self.assertIn('discarded', categorized)
        self.assertEqual(len(categorized['discarded']), 1)
        logger.info("categorize_datasets_by_project test passed.")

        # Test group_datasets_by_project_id
        logger.info("Testing group_datasets_by_project_id...")
        grouped = group_datasets_by_project_id(categorized)
        self.assertIn('PRJNA', grouped)
        self.assertIn('PRJNA1', grouped['PRJNA'])
        self.assertEqual(len(grouped['PRJNA']['PRJNA1']), 1)
        logger.info("group_datasets_by_project_id test passed.")

        # Test get_project_statistics
        logger.info("Testing get_project_statistics...")
        stats = get_project_statistics(categorized)
        self.assertEqual(stats['total_records'], 6)
        self.assertEqual(stats['categories']['GSE']['count'], 1)
        logger.info("get_project_statistics test passed.")

        # Test create_classify_ready_export
        logger.info("Testing create_classify_ready_export...")
        export_ready = create_classify_ready_export(categorized)
        self.assertIn('metadata', export_ready)
        self.assertIn('categorized_data', export_ready)
        self.assertIn('grouped_data', export_ready)
        self.assertIn('GSE', export_ready['categorized_data'])
        self.assertIn('PRJNA', export_ready['categorized_data'])
        logger.info("create_classify_ready_export test passed.")

        # Test run_export_workflow
        logger.info("Testing run_export_workflow...")
        mock_conn = MagicMock()

        with patch('SRAgent.db.export.json_export.get_prefiltered_datasets_functional', side_effect=lambda *args, **kwargs: df):
            workflow_result = run_export_workflow(mock_conn, output_path=self.output_json_path)
            self.assertEqual(workflow_result["status"], "success")
            self.assertEqual(workflow_result["output_path"], self.output_json_path)
            logger.info(f"run_export_workflow test passed. Exported to {self.output_json_path}")

            # Verify the file content
            with open(self.output_json_path, 'r') as f:
                exported_content = json.load(f)
                self.assertIn('timestamp', exported_content['export_metadata'])
                self.assertIn('total_records', exported_content['export_metadata'])
                self.assertEqual(exported_content['export_metadata']['total_records'], len(data['sra_id']))
                self.assertIn('filter_parameters', exported_content['export_metadata'])
                self.assertIn('categorization_stats', exported_content['export_metadata'])
                self.assertIn('grouping_stats', exported_content['export_metadata'])
                self.assertEqual(len(exported_content['raw_data']), len(data['sra_id']))

            logger.info("Exported file content verified.")

        logger.info("--- All categorize module tests passed! ---")

if __name__ == '__main__':
    unittest.main()