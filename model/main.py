import os
from typing import Dict, Any, List
import json
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import requests
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from statsmodels.tsa.arima.model import ARIMA
from dotenv import load_dotenv
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
            region='us-west-2'
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

        # Handle missing values
        data_imputed = self.imputer.fit_transform(data[['value']])
        
        # Perform advanced EDA
        stats = pd.Series(data_imputed.flatten()).describe()
        trend = 'increasing' if data_imputed[-1] > data_imputed[0] else 'decreasing'
        
        # Perform clustering
        scaled_data = self.scaler.fit_transform(data_imputed)
        n_clusters = min(3, len(data))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        
        try:
            data['cluster'] = kmeans.fit_predict(scaled_data)
            clusters = data['cluster'].value_counts().to_dict()
        except Exception as e:
            print(f"Clustering failed: {str(e)}")
            clusters = {"error": str(e)}
        
        # Perform time series forecasting
        try:
            model = ARIMA(data_imputed.flatten(), order=(1,0,0))
            results = model.fit()
            forecast = results.forecast(steps=5)
        except Exception as e:
            print(f"ARIMA forecasting failed: {str(e)}")
            forecast = np.array([data_imputed.mean()] * 5)  # fallback to mean if ARIMA fails

        analysis_result = {
            "basic_stats": stats.to_dict(),
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
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""
        Based on the following analysis, suggest actions to take:
        {json.dumps(data)}
        
        Provide your response in the following JSON format:
        {{
            "recommended_action": "Brief description of the main action to take",
            "justification": "Explanation of why this action is recommended",
            "alternative_actions": ["List", "of", "alternative", "actions"],
            "potential_risks": ["List", "of", "potential", "risks"],
            "next_steps": ["List", "of", "next", "steps", "to", "implement", "the", "action"]
        }}
        """
        ai_decision = self.call_gemini_api(prompt)
        
        # Remove any Markdown code block indicators
        ai_decision = ai_decision.replace('```json', '').replace('```', '').strip()
        
        try:
            decision = json.loads(ai_decision)
        except json.JSONDecodeError:
            print(f"Error: Gemini API response is not a valid JSON string: {ai_decision}")
            decision = {
                "recommended_action": "Take no action",
                "justification": "The response from the Gemini API is not in the expected format.",
                "alternative_actions": [],
                "potential_risks": [],
                "next_steps": []
            }
        
        return decision

class MiniZapWithEDA:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.agents = {
            "collector": DataCollectorAgent("DataCollector", "Collects and enriches data"),
            "analyst": AnalystAgent("Analyst", "Performs advanced EDA"),
            "decision_maker": DecisionMakerAgent("DecisionMaker", "Makes strategic decisions based on analysis")
        }

    def trigger(self, payloads: List[Dict[str, Any]]):
        print("Triggers received:")
        print(json.dumps(payloads, indent=2))
        enriched_data = self.agents["collector"].process(payloads)
        
        # Store in Pinecone
        for data in enriched_data:
            vector = self.model.encode([json.dumps(data)])[0].tolist()
            try:
                index.upsert([(str(datetime.now().timestamp()), vector, data)])
            except Exception as e:
                print(f"Error upserting enriched data to Pinecone: {str(e)}")
        
        return enriched_data

    def process_data(self):
        # Query all vectors from Pinecone
        query_response = index.query(vector=[0]*384, top_k=1000, include_metadata=True)
        
        data = [match['metadata'] for match in query_response['matches']]
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df

    def perform_eda(self, df: pd.DataFrame):
        try:
            analysis_result = self.agents["analyst"].process(df)
            
            plt.figure(figsize=(10, 6))
            df['value'].plot()
            plt.title('Value over Time')
            plt.xlabel('Timestamp')
            plt.ylabel('Value')
            plt.savefig('value_over_time.png')
            plt.close()
            
            return analysis_result
        except Exception as e:
            print(f"EDA failed: {str(e)}")
            return {"error": str(e)}

    def take_action(self, analysis_result: Dict[str, Any]):
        decision = self.agents["decision_maker"].process(analysis_result)
        print(f"Decision made: {json.dumps(decision, indent=2)}")
        return decision

    def run_flow(self, payloads: List[Dict[str, Any]]):
        enriched_data = self.trigger(payloads)
        df = self.process_data()
        analysis_result = self.perform_eda(df)
        decision = self.take_action(analysis_result)
        
        flow_result = {
            "timestamp": datetime.now().isoformat(),
            "enriched_data_summary": f"Processed {len(enriched_data)} data points",
            "analysis_result_summary": json.dumps({
                "basic_stats": analysis_result.get("basic_stats", {}),
                "trend": analysis_result.get("trend", ""),
                "clusters": analysis_result.get("clusters", {})
            }),
            "decision_summary": json.dumps({
                "recommended_action": decision.get("recommended_action", ""),
                "justification": decision.get("justification", "")
            })
        }
        
        # Debug: Print flow_result structure and size
        print("Flow result structure:")
        print(json.dumps(flow_result, indent=2))
        print(f"Flow result size: {len(json.dumps(flow_result))} bytes")

        vector = self.model.encode([json.dumps(flow_result)])[0].tolist()
        
        # Debug: Check vector dimension
        print(f"Vector dimension: {len(vector)}")

        try:
            index.upsert([(str(datetime.now().timestamp()), vector, flow_result)])
            print("Successfully upserted flow result to Pinecone")
        except Exception as e:
            print(f"Error upserting flow result to Pinecone: {str(e)}")
            # Try upserting with minimal data
            minimal_flow_result = {"timestamp": flow_result["timestamp"]}
            try:
                index.upsert([(str(datetime.now().timestamp()), vector, minimal_flow_result)])
                print("Successfully upserted minimal data to Pinecone")
            except Exception as e:
                print(f"Error upserting minimal data to Pinecone: {str(e)}")

# Example usage
if __name__ == "__main__":
    mini_zap = MiniZapWithEDA()

    # Generate a batch of data points
    batch_size = 10
    start_time = datetime.now()
    batch_data = [
        {
            "timestamp": (start_time + timedelta(minutes=i)).isoformat(),
            "value": 40 + i * 5,
            "source": f"sensor_{i % 3}"
        }
        for i in range(batch_size)
    ]

    # Run the flow with the batch of data points
    mini_zap.run_flow(batch_data)

    print("\nFinal EDA Results and Decision:")
    final_df = mini_zap.process_data()
    final_analysis = mini_zap.perform_eda(final_df)
    final_decision = mini_zap.take_action(final_analysis)
    
    print("\nAnalysis Result:")
    print(json.dumps(final_analysis, indent=2))
    print("\nFinal Decision:")
    print(json.dumps(final_decision, indent=2))
    print("\nA plot 'value_over_time.png' has been saved in the current directory.")

    # Retrieve and display flow results
    query_response = index.query(vector=[0]*384, top_k=1, include_metadata=True)
    print("\nLast Flow Result:")
    flow_result = query_response['matches'][0]['metadata']
    
    print(json.dumps(flow_result, indent=2))
