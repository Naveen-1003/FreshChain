"""
Lightweight AI Suggestion System - Fallback for when Phi model isn't available
Provides rule-based intelligent suggestions for retailers
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import random

logger = logging.getLogger(__name__)

class LightweightAIService:
    """Lightweight rule-based AI service for retail suggestions"""
    
    def __init__(self):
        self.is_loaded = True  # Always available
        
    def generate_inventory_suggestions(self, inventory_data: Dict) -> List[str]:
        """Generate rule-based suggestions for inventory optimization"""
        try:
            suggestions = []
            batches = inventory_data.get('batches', [])
            env_data = inventory_data.get('environmental_data', {})
            
            total_items = len(batches)
            expiring_soon = len([b for b in batches 
                               if b.get('shelf_life_prediction', {}).get('predicted_days', 10) < 3])
            
            temp = env_data.get('temperature', 20)
            humidity = env_data.get('humidity', 60)
            
            # Temperature-based suggestions
            if temp > 25:
                suggestions.append("🌡️ High temperature detected ({}°C). Consider moving perishable items to cooler storage areas to extend shelf life.".format(temp))
            elif temp < 10:
                suggestions.append("❄️ Low temperature detected ({}°C). Some fruits may lose quality - monitor for cold damage.".format(temp))
            
            # Humidity-based suggestions
            if humidity > 80:
                suggestions.append("💧 High humidity ({}%) detected. Increase ventilation to prevent mold and spoilage.".format(humidity))
            elif humidity < 40:
                suggestions.append("🏜️ Low humidity ({}%) may cause dehydration in fresh produce. Consider humidification.".format(humidity))
            
            # Inventory level suggestions
            if total_items > 50:
                suggestions.append("📦 High inventory levels detected. Consider promotional pricing to move stock faster.")
            elif total_items < 10:
                suggestions.append("⚠️ Low inventory levels. Plan restocking to avoid stockouts.")
            
            # Expiry-based suggestions
            if expiring_soon > 0:
                suggestions.append("⏰ {} items expiring within 3 days. Implement discount strategy or move to quick-sale section.".format(expiring_soon))
            
            # Product mix suggestions
            product_types = set(b.get('product_name', '').lower() for b in batches)
            if 'apple' in product_types and 'banana' in product_types:
                suggestions.append("🍎🍌 Cross-sell opportunity: Bundle apples and bananas for healthy snack packs.")
            
            # Seasonal suggestions
            month = datetime.now().month
            if month in [6, 7, 8]:  # Summer months
                suggestions.append("☀️ Summer season: Promote cold beverages and fresh fruits for increased sales.")
            elif month in [12, 1, 2]:  # Winter months
                suggestions.append("❄️ Winter season: Focus on preserved items and root vegetables with longer shelf life.")
            
            # Quality-based suggestions
            poor_quality_items = len([b for b in batches 
                                    if float(b.get('freshness_prediction', 1)) < 0.5])
            if poor_quality_items > 0:
                suggestions.append("🔍 {} items showing quality concerns. Implement first-in-first-out rotation.".format(poor_quality_items))
            
            # Default suggestions if none generated
            if not suggestions:
                suggestions = [
                    "📊 Inventory levels look good. Continue monitoring environmental conditions.",
                    "✅ Maintain current storage practices for optimal freshness retention.",
                    "📈 Consider analyzing sales patterns to optimize ordering schedules."
                ]
            
            return suggestions[:5]  # Return top 5 suggestions
            
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            return ["Unable to generate suggestions at this time. Please check your inventory data."]
    
    def generate_chat_response(self, message: str, context: Dict = None) -> str:
        """Generate rule-based chat responses"""
        try:
            message_lower = message.lower()
            
            # Inventory-related queries
            if any(word in message_lower for word in ['inventory', 'stock', 'items']):
                if context and 'batches' in context:
                    total_items = len(context['batches'])
                    return f"You currently have {total_items} items in inventory. I recommend maintaining optimal storage conditions and implementing FIFO rotation for best results."
                return "To help with inventory management, ensure you're tracking expiry dates and maintaining proper storage conditions."
            
            # Temperature-related queries
            elif any(word in message_lower for word in ['temperature', 'temp', 'hot', 'cold']):
                if context and 'environmental_data' in context:
                    temp = context['environmental_data'].get('temperature', 'N/A')
                    return f"Current temperature is {temp}°C. For most fresh produce, optimal storage temperature is between 2-8°C for refrigerated items and 15-20°C for ambient storage."
                return "Temperature control is crucial for freshness. Most perishables should be kept between 2-8°C, while dry goods can be stored at room temperature (15-20°C)."
            
            # Humidity-related queries
            elif any(word in message_lower for word in ['humidity', 'moisture', 'damp']):
                if context and 'environmental_data' in context:
                    humidity = context['environmental_data'].get('humidity', 'N/A')
                    return f"Current humidity is {humidity}%. Optimal humidity for most fresh produce is 85-95% for leafy greens and 80-85% for fruits."
                return "Humidity management is important. Most fresh produce requires high humidity (80-95%) while dry goods need low humidity (below 65%) to prevent spoilage."
            
            # Expiry-related queries
            elif any(word in message_lower for word in ['expiry', 'expire', 'shelf life', 'spoil']):
                return "To manage expiry dates effectively: 1) Implement FIFO rotation, 2) Monitor environmental conditions, 3) Use discount strategies for items nearing expiry, 4) Track patterns to optimize ordering."
            
            # Sales-related queries
            elif any(word in message_lower for word in ['sales', 'profit', 'revenue', 'pricing']):
                return "For better sales performance: 1) Analyze demand patterns, 2) Implement dynamic pricing for items nearing expiry, 3) Create bundles and cross-sell opportunities, 4) Monitor competitor pricing."
            
            # Quality-related queries
            elif any(word in message_lower for word in ['quality', 'fresh', 'rotten', 'bad']):
                return "Quality management tips: 1) Regular visual inspections, 2) Proper storage conditions, 3) FIFO rotation, 4) Separate damaged items immediately, 5) Train staff on quality indicators."
            
            # General business queries
            elif any(word in message_lower for word in ['business', 'improve', 'optimize', 'better']):
                return "Key optimization strategies: 1) Monitor real-time environmental data, 2) Implement predictive analytics for demand, 3) Optimize storage conditions, 4) Use data-driven pricing strategies, 5) Focus on waste reduction."
            
            # Greeting responses
            elif any(word in message_lower for word in ['hello', 'hi', 'help']):
                return "Hello! I'm your retail assistant. I can help you with inventory management, quality control, pricing strategies, and operational optimization. What would you like to know?"
            
            # Default response
            else:
                return "I'm here to help with your retail operations. You can ask me about inventory management, quality control, environmental conditions, pricing strategies, or general business optimization."
                
        except Exception as e:
            logger.error(f"Error generating chat response: {e}")
            return "I apologize, but I'm having trouble processing your request. Please try rephrasing your question."
    
    def analyze_trends(self, analytics_data: Dict) -> Dict[str, Any]:
        """Analyze business trends with rule-based logic"""
        try:
            insights = []
            
            sales_trend = analytics_data.get('sales_trend', 'stable')
            inventory_turnover = analytics_data.get('inventory_turnover', 'average')
            quality_issues = analytics_data.get('quality_issues', 0)
            
            # Sales trend analysis
            if sales_trend == 'increasing':
                insights.append("📈 Sales trending upward - consider increasing inventory levels and exploring expansion opportunities.")
            elif sales_trend == 'decreasing':
                insights.append("📉 Sales declining - review pricing strategy and product mix. Consider promotional campaigns.")
            else:
                insights.append("📊 Sales stable - maintain current strategies while looking for growth opportunities.")
            
            # Inventory turnover analysis
            if inventory_turnover == 'high':
                insights.append("🔄 High inventory turnover indicates good demand forecasting and efficient operations.")
            elif inventory_turnover == 'low':
                insights.append("⚠️ Low inventory turnover - review product selection and consider markdown strategies.")
            
            # Quality issues analysis
            if quality_issues > 5:
                insights.append("🔍 Multiple quality issues detected - review storage conditions and supplier quality.")
            elif quality_issues == 0:
                insights.append("✅ No quality issues - excellent quality management practices.")
            
            return {
                "insights": " ".join(insights),
                "confidence": 0.75,
                "timestamp": datetime.now().isoformat(),
                "recommendations": [
                    "Monitor environmental conditions regularly",
                    "Implement data-driven inventory management",
                    "Focus on quality control processes"
                ]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
            return {"error": "Unable to analyze trends at this time"}

# Global lightweight service instance
lightweight_ai = LightweightAIService()

# Fallback functions for when Phi model is not available
def get_lightweight_suggestions(inventory_data: Dict) -> List[str]:
    """Get lightweight AI suggestions"""
    return lightweight_ai.generate_inventory_suggestions(inventory_data)

def chat_with_lightweight_ai(message: str, context: Dict = None) -> str:
    """Chat with lightweight AI"""
    return lightweight_ai.generate_chat_response(message, context)

def analyze_trends_lightweight(analytics_data: Dict) -> Dict[str, Any]:
    """Analyze trends with lightweight AI"""
    return lightweight_ai.analyze_trends(analytics_data)