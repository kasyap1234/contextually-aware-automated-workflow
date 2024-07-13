import os
from typing import Dict, Any, List
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from statsmodels.tsa.arima.model import ARIMA
from dotenv import load_dotenv
import random

load_dotenv()

# Environment setup
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "mini-zap-index"

# Initialize Pinecone
pinecone = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
if PINECONE_INDEX_NAME not in pinecone.list_indexes().names():
    pinecone.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

# Get the index instance
index = pinecone.Index(PINECONE_INDEX_NAME)

class AIAgent:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role

    def call_gemini_api(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        data = {"contents": [{"parts":[{"text": prompt}]}]}
        response = requests.post(GEMINI_API_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()['candidates'][0]['content']['parts'][0]['text']

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Each agent should implement its own process method")

class DataCollectorAgent(AIAgent):
    def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        enriched_data = []
        for item in data:
            prompt = f"Analyze the following data and suggest any additional fields or enrichments that could be valuable: {json.dumps(item)}"
            suggestion = self.call_gemini_api(prompt)
            item_copy = item.copy()
            item_copy['agent_suggestions'] = suggestion
            enriched_data.append(item_copy)
        return enriched_data

class AnalystAgent(AIAgent):
    def __init__(self, name: str, role: str):
        super().__init__(name, role)
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')

    def process(self, data: pd.DataFrame) -> Dict[str, Any]:
        # Ensure index is sorted and has a frequency
        data = data.sort_index()
        data.index = pd.DatetimeIndex(data.index).to_period('D')

        # Select numeric columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns

        # Handle missing values and scale data
        data_imputed = self.imputer.fit_transform(data[numeric_columns])
        data_scaled = self.scaler.fit_transform(data_imputed)

        # Perform advanced EDA
        stats = pd.DataFrame(data_imputed, columns=numeric_columns).describe().to_dict()
        
        # Calculate overall trend
        trend = 'increasing' if np.mean(data_imputed[-1] > data_imputed[0]) else 'decreasing'
        
        # Perform clustering
        n_clusters = min(3, len(data))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        
        try:
            data['cluster'] = kmeans.fit_predict(data_scaled)
            clusters = data['cluster'].value_counts().to_dict()
        except Exception as e:
            print(f"Clustering failed: {str(e)}")
            clusters = {"error": str(e)}
        
        # Perform time series forecasting (using the first numeric column as an example)
        try:
            model = ARIMA(data_imputed[:, 0], order=(1,0,0))
            results = model.fit()
            forecast = results.forecast(steps=5)
        except Exception as e:
            print(f"ARIMA forecasting failed: {str(e)}")
            forecast = np.array([data_imputed[:, 0].mean()] * 5)  # fallback to mean if ARIMA fails

        analysis_result = {
            "basic_stats": stats,
            "trend": trend,
            "clusters": clusters,
            "forecast": forecast.tolist()
        }
        
        # Get AI interpretation
        prompt = f"Interpret the following analysis results and provide insights: {json.dumps(analysis_result)}"
        ai_interpretation = self.call_gemini_api(prompt)
        
        analysis_result['ai_interpretation'] = ai_interpretation
        return analysis_result

class DecisionMakerAgent(AIAgent):
    def __init__(self, name: str, role: str):
        super().__init__(name, role)
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1

    def get_state(self, data):
        # Simplify the state representation
        return tuple(sorted(data.keys()))

    def get_action(self, state):
        if random.random() < self.epsilon or state not in self.q_table:
            return self.generate_random_action()
        else:
            return max(self.q_table[state], key=self.q_table[state].get)

    def generate_random_action(self):
        actions = ["increase_marketing", "reduce_costs", "expand_product_line", "improve_customer_service"]
        return random.choice(actions)

    def update_q_table(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in ["increase_marketing", "reduce_costs", "expand_product_line", "improve_customer_service"]}
        
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table.get(next_state, {}).values(), default=0)
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        state = self.get_state(data)
        action = self.get_action(state)
        
        prompt = f"""
        Based on the following analysis and the chosen action {action}, suggest detailed steps to take:
        {json.dumps(data)}
        
        Your response MUST be in the following JSON format, with no additional text or markdown formatting:
        {{
            "recommended_action": "{action}",
            "justification": "Explanation of why this action is recommended",
            "detailed_steps": ["Step 1", "Step 2", "Step 3"],
            "potential_risks": ["Risk 1", "Risk 2", "Risk 3"],
            "expected_outcomes": ["Outcome 1", "Outcome 2", "Outcome 3"]
        }}
        """
        ai_decision = self.call_gemini_api(prompt)
        print("Gemini API response:", ai_decision)
        
        # Remove markdown code block indicators if present
        ai_decision = ai_decision.strip().removeprefix("```json").removesuffix("```")
        
        try:
            decision = json.loads(ai_decision)
            # Validate the structure of the decision
            required_keys = ["recommended_action", "justification", "detailed_steps", "potential_risks", "expected_outcomes"]
            if not all(key in decision for key in required_keys):
                raise ValueError("Response is missing required keys")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing Gemini API response: {str(e)}. Falling back to default structure.")
            decision = {
                "recommended_action": action,
                "justification": "Unable to parse API response",
                "detailed_steps": ["Review API response", "Adjust prompt if necessary", "Retry decision-making process"],
                "potential_risks": ["Incomplete decision information", "Potential misalignment with data analysis"],
                "expected_outcomes": ["Improved API response parsing", "More accurate decision-making"]
            }
        
        return decision

        
        
        try:
            decision = json.loads(ai_decision)
            # Validate the structure of the decision
            required_keys = ["recommended_action", "justification", "detailed_steps", "potential_risks", "expected_outcomes"]
            if not all(key in decision for key in required_keys):
                raise ValueError("Response is missing required keys")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing Gemini API response: {str(e)}. Falling back to default structure.")
            decision = {
                "recommended_action": action,
                "justification": "Unable to parse API response",
                "detailed_steps": ["Review API response", "Adjust prompt if necessary", "Retry decision-making process"],
                "potential_risks": ["Incomplete decision information", "Potential misalignment with data analysis"],
                "expected_outcomes": ["Improved API response parsing", "More accurate decision-making"]
            }
        
        return decision


    def extract_text_between(self, text, start_marker, end_marker):
        start = text.find(start_marker)
        end = text.find(end_marker)
        if start != -1 and end != -1:
            return text[start + len(start_marker):end].strip()
        return ""

    def extract_list(self, text, marker):
        start = text.find(marker)
        if start != -1:
            list_text = text[start + len(marker):]
            return [item.strip() for item in list_text.split('\n') if item.strip().startswith('-')]
        return []

class MiniZapWithEDA:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.agents = {
            "collector": DataCollectorAgent("DataCollector", "Collects and enriches data"),
            "analyst": AnalystAgent("Analyst", "Performs advanced EDA"),
            "decision_maker": DecisionMakerAgent("DecisionMaker", "Makes strategic decisions based on analysis")
        }
        self.previous_state = None
        self.previous_action = None

    def trigger(self, payloads: List[Dict[str, Any]]):
        print("Triggers received:")
        print(json.dumps(payloads, indent=2))
        enriched_data = self.agents["collector"].process(payloads)
        
        for data in enriched_data:
            vector = self.model.encode([json.dumps(data)])[0].tolist()
            try:
                index.upsert([(str(datetime.now().timestamp()), vector, data)])
            except Exception as e:
                print(f"Error upserting enriched data to Pinecone: {str(e)}")
        
        return enriched_data

    def process_data(self):
        query_response = index.query(vector=[0]*384, top_k=1000, include_metadata=True)
        data = [match['metadata'] for match in query_response['matches']]
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df

    def perform_eda(self, df: pd.DataFrame):
        try:
            analysis_result = self.agents["analyst"].process(df)
            
            eda_results = {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "data_types": df.dtypes.to_dict(),
                "summary_statistics": df.describe().to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "correlation_matrix": df.corr().to_dict() if df.select_dtypes(include=[np.number]).shape[1] > 1 else None
            }
            
            if isinstance(df.index, pd.DatetimeIndex):
                eda_results["time_range"] = {
                    "start": df.index.min().isoformat(),
                    "end": df.index.max().isoformat(),
                    "duration": str(df.index.max() - df.index.min())
                }
            
            analysis_result["eda_results"] = eda_results
            return analysis_result
        except Exception as e:
            print(f"EDA failed: {str(e)}")
            return {"error": str(e)}

    def take_action(self, analysis_result: Dict[str, Any]):
        decision = self.agents["decision_maker"].process(analysis_result)
        print(f"Decision made: {json.dumps(decision, indent=2)}")
        return decision

    def provide_feedback(self, reward: float):
        if self.previous_state and self.previous_action:
            current_state = self.agents["decision_maker"].get_state(self.process_data().to_dict())
            self.agents["decision_maker"].update_q_table(
                self.previous_state, 
                self.previous_action, 
                reward, 
                current_state
            )
        else:
            print("No previous state or action to update.")

    def run_flow(self, payloads: List[Dict[str, Any]]):
        enriched_data = self.trigger(payloads)
        df = self.process_data()
        analysis_result = self.perform_eda(df)
        decision = self.take_action(analysis_result)
        
        current_state = self.agents["decision_maker"].get_state(df.to_dict())
        self.previous_state = current_state
        self.previous_action = decision["recommended_action"]
        
        flow_result = {
            "timestamp": datetime.now().isoformat(),
            "enriched_data_summary": f"Processed {len(enriched_data)} data points",
            "analysis_result_summary": json.dumps({
                "basic_stats": analysis_result.get("basic_stats", {}),
                "trend": analysis_result.get("trend", ""),
                "clusters": analysis_result.get("clusters", {}),
                "forecast": analysis_result.get("forecast", [])
            }),
            "decision_summary": json.dumps(decision)
        }
        
        vector = self.model.encode([json.dumps(flow_result)])[0].tolist()
        
        try:
            index.upsert([(str(datetime.now().timestamp()), vector, flow_result)])
        except Exception as e:
            print(f"Error upserting flow result to Pinecone: {str(e)}")
            print("Attempting to upsert with simplified metadata...")
            simplified_flow_result = {k: str(v) for k, v in flow_result.items()}
            try:
                index.upsert([(str(datetime.now().timestamp()), vector, simplified_flow_result)])
            except Exception as e:
                print(f"Error upserting simplified flow result to Pinecone: {str(e)}")

        return flow_result


if __name__ == "__main__":
    mini_zap = MiniZapWithEDA()

    batch_size = 15
    start_time = datetime.now()
    batch_data = [
        {
            "timestamp": (start_time + timedelta(days=i)).isoformat(),
            "revenue": 100000 + i * 5000,
            "expenses": 80000 + i * 3000,
            "profit_margin": round((100000 + i * 5000 - (80000 + i * 3000)) / (100000 + i * 5000), 2),
            "customer_acquisition_cost": 50 + i * 0.5,
            "customer_lifetime_value": 500 + i * 10,
            "churn_rate": round(0.05 - i * 0.002, 3),
            "active_users": 10000 + i * 100,
            "average_order_value": 100 + i * 2,
            "conversion_rate": round(0.03 + i * 0.001, 3),
            "customer_satisfaction_score": round(4.0 + i * 0.05, 1),
            "website_traffic": 50000 + i * 1000,
             "social_media_engagement": 5000 + i * 200,
            "product_inventory": 1000 - i * 20,
            "employee_productivity": round(0.8 + i * 0.01, 2),
            "source": f"department_{i % 5}"
        }
        for i in range(batch_size)
    ]
    
    flow_result = mini_zap.run_flow(batch_data)

    print("\nFinal Flow Result:")
    print(json.dumps(flow_result, indent=2))

    # Simulate feedback
    feedback_reward = 0.5  # This could be based on actual business outcomes
    mini_zap.provide_feedback(feedback_reward)

    print("\nRetrieving last flow result from Pinecone:")
    query_response = index.query(vector=[0]*384, top_k=1, include_metadata=True)
    last_flow_result = query_response['matches'][0]['metadata']
    print(json.dumps(last_flow_result, indent=2))
