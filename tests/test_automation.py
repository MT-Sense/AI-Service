import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.automation import router, ProposalRequest, ProposalResponse, validate_result


class AutomationTests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(router)
        self.client = TestClient(app)
        self.payload = {'candidates': [{'playbook': 'training', 'evidence': [
            {'id': 'survey', 'source': 'survey', 'text': '5 responses mention growth'}],
            'samples': ['อยากเรียนรู้เพิ่ม'], 'missingData': ['No catalog']} ]}
        self.result = {'drafts': [{'playbook': 'training', 'problemFound': True,
            'title': 'สำรวจความต้องการอบรม', 'rationale': 'มีความต้องการเรียนรู้เพิ่ม',
            'draft': 'สอบถามหลักสูตรและงบประมาณก่อนเลือก', 'evidenceIds': ['survey'],
            'missingData': ['No catalog']}]}

    def test_auth_before_model(self):
        with patch.dict(os.environ, {'AUTOMATION_SERVICE_TOKEN': 'secret'}), patch('app.automation.client.models.generate_content') as model:
            self.assertEqual(self.client.post('/automation/propose', json=self.payload).status_code, 403)
            model.assert_not_called()

    def test_structured_response_and_separate_system_prompt(self):
        with patch.dict(os.environ, {'AUTOMATION_SERVICE_TOKEN': 'secret'}), patch('app.automation.client.models.generate_content', return_value=SimpleNamespace(text=json.dumps(self.result))) as model:
            r = self.client.post('/automation/propose', json=self.payload, headers={'X-Automation-Token': 'secret'})
            self.assertEqual(r.status_code, 200, r.text)
            args = model.call_args.kwargs
            self.assertIn('UNTRUSTED DATA', args['config']['system_instruction'])
            self.assertNotIn('tools', args['config'])
            self.assertEqual(r.json(), self.result)

    def test_fabricated_evidence_is_rejected(self):
        self.result['drafts'][0]['evidenceIds'] = ['imaginary-jira']
        with self.assertRaises(ValueError):
            validate_result(ProposalRequest(**self.payload), ProposalResponse(**self.result))

    def test_positive_only_no_problem_is_valid(self):
        self.result['drafts'][0].update(problemFound=False, title='', draft='', evidenceIds=[])
        self.assertFalse(validate_result(ProposalRequest(**self.payload), ProposalResponse(**self.result)).drafts[0].problemFound)

    def test_omission_and_duplicate_rejected(self):
        for drafts in [[], self.result['drafts'] * 2]:
            with self.assertRaises(ValueError):
                validate_result(ProposalRequest(**self.payload), ProposalResponse(drafts=drafts))

    def test_model_failure_is_explicit(self):
        with patch.dict(os.environ, {'AUTOMATION_SERVICE_TOKEN': 'secret'}), patch('app.automation.client.models.generate_content', side_effect=RuntimeError('secret prompt')):
            r = self.client.post('/automation/propose', json=self.payload, headers={'X-Automation-Token': 'secret'})
            self.assertEqual(r.status_code, 502)
            self.assertNotIn('secret prompt', r.text)
