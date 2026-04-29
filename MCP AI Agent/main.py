import asyncio
import time
import logging
import re
from typing import Dict, Any
from utils.env import load_env
from agents.planner_agent import build_planner_agent, classify_target
from agents.hr_agent import build_hr_agent
from agents.compliance_agent import build_compliance_agent
from agents.finance_agent import build_finance_agent
from tools.azure_search_tool import AzureSearchTool

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def run_multi_agent(query: str, agents: Dict[str, Any]) -> Dict[str, Any]:
   """
   Advanced multi-agent system with routing, search context, and ticket creation capabilities.
   """
   start_time = time.time()

   try:
      # Step 1: Route the query
      logging.info(f"Routing query: {query[:50]}...")
      target = await classify_target(agents["planner"], query)
      logging.info(f"Query routed to: {target}")

      # Step 2: Retrieve relevant context using Azure Search
      logging.info("Retrieving context from knowledge base...")
      context = await agents["search_tool"].search(query, top=3)

      # Step 3: Create enriched prompt with context
      enriched_prompt = f"""
Context from Knowledge Base:
{context}

---

User Question: {query}

Please provide a comprehensive answer based PRIMARILY on the context information provided above. 
Use the knowledge base content as your primary source of truth. If the context contains relevant 
information, base your answer on that. Only supplement with general knowledge if the context 
doesn't cover the specific question.

If no relevant information is found in the context, clearly state that and provide general guidance 
while recommending the user contact the appropriate department for specific details.
"""

      # Step 4: Get response from appropriate agent
      agent_mapping = {
            "HR": ("hr", "HRAgent"),
            "FINANCE": ("finance", "FinanceAgent"), 
            "COMPLIANCE": ("compliance", "ComplianceAgent")
      }

      if target in agent_mapping:
            agent_key, agent_name = agent_mapping[target]
            answer = await agents[agent_key].run(enriched_prompt)
      else:
            # Fallback to HR if routing unclear
            logging.warning(f"Unknown target '{target}', falling back to HR")
            answer = await agents["hr"].run(enriched_prompt)
            target = "HR"
            agent_name = "HRAgent"

      answer_text = str(answer)

      # Step 6: Process response
      response_time = time.time() - start_time

      return {
            "query": query,
            "routed_to": target,
            "agent_name": agent_name,
            "answer": answer_text,
            "context_retrieved": len(context) > 100,  # Simple check if context was found
            "ticket_created": False,  # Tickets only created in interactive mode
            "ticket_info": None,
            "response_time": round(response_time, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "success": True
      }

   except Exception as e:
      logging.error(f"Error processing query: {e}")
      return {
            "query": query,
            "routed_to": "ERROR",
            "agent_name": "ErrorHandler",
            "answer": f"I apologize, but I encountered an error processing your request: {str(e)}",
            "context_retrieved": False,
            "ticket_created": False,
            "ticket_info": None,
            "response_time": round(time.time() - start_time, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "success": False
      }

def format_response(result: Dict[str, Any]) -> str:
   """Format the agent response for display."""
   status_icon = "✅" if result["success"] else "❌"
   context_icon = "📚" if result.get("context_retrieved") else "📭"
   ticket_icon = "🎫" if result.get("ticket_created") else ""

   formatted = f"""
{status_icon} Agent Response Summary:
┌─ Routed to: {result['routed_to']} ({result['agent_name']})
├─ Response time: {result['response_time']}s
├─ Context retrieved: {context_icon} {'Yes' if result.get('context_retrieved') else 'No'}
├─ Ticket created: {ticket_icon} {'Yes' if result.get('ticket_created') else 'No'}
├─ Timestamp: {result['timestamp']}
└─ Status: {'Success' if result['success'] else 'Error'}

💬 Answer:
{result['answer']}
"""

   # Add ticket details if available
   if result.get("ticket_info") and result["ticket_info"].get("success"):
      ticket = result["ticket_info"]["ticket"]
      formatted += f"""

🎫 Ticket Details:
├─ ID: #{ticket['id']}
├─ Status: {ticket['status']}
├─ Priority: {ticket['priority']}
└─ URL: {ticket['url']}
"""

   return formatted

async def run_interactive_mode(agents: Dict[str, Any]):
   """Interactive mode for real-time queries."""
   print("\n🤖 Enterprise Agent System - Interactive Mode")
   print("Available agents: HR, Finance, Compliance")
   print("Type 'quit' to exit, 'help' for commands\n")

   while True:
      try:
            query = input("Enter your question: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
               print("👋 Goodbye!")
               break
            elif query.lower() == 'help':
               print("""
📋 Available Commands:
- Ask any question about HR, Finance, or Compliance
- 'quit' or 'exit' - Exit the system
- 'help' - Show this help message

🎯 Example questions:
- "What's the travel reimbursement limit for meals?"
- "How many vacation days do employees get?"  
- "Do we need GDPR compliance for EU customers?"
""")
               continue
            elif not query:
               continue

            result = await run_multi_agent(query, agents)
            print(format_response(result))

      except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
      except Exception as e:
            logging.error(f"Interactive mode error: {e}")
            print(f"❌ Error: {e}")

async def run_batch_tests(agents: Dict[str, Any]):
   """Run focused test queries with grounded data integration."""
   test_queries = [
      "What is the travel reimbursement limit for hotel stays?",
      "How many vacation days are allowed per year?"
   ]

   print("🧪 Running focused batch tests with grounded data integration...\n")

   for i, query in enumerate(test_queries, 1):
      print(f"{'='*80}")
      print(f"TEST {i}/{len(test_queries)}: {query}")
      print(f"{'='*80}")

      result = await run_multi_agent(query, agents)
      print(format_response(result))

      # Small delay between queries for better readability
      if i < len(test_queries):
            await asyncio.sleep(1.0)  # Longer delay for tool operations

async def main():
   """Main application entry point with enhanced features and tool integration."""
   print("🚀 Initializing Enterprise Agent System with Tools...")

   try:
      # Load environment and build agents
      load_env()
      logging.info("Building agent network...")

      # Build core agents
      agents = {
            "planner": await build_planner_agent(),
            "hr": await build_hr_agent(), 
            "compliance": await build_compliance_agent(),
            "finance": await build_finance_agent()
      }

      # Initialize and attach tools
      logging.info("Initializing tools...")

      try:
            search_tool = AzureSearchTool()
            agents["search_tool"] = search_tool

            # Test search tool
            health = await search_tool.health_check()
            if health["status"] == "healthy":
               logging.info("✅ Azure Search tool initialized successfully")
            else:
               logging.warning(f"⚠️ Azure Search tool health check failed: {health}")

      except Exception as e:
            logging.error(f"Failed to initialize Azure Search tool: {e}")
            # Create mock search tool for testing
            class MockSearchTool:
               async def search(self, query, top=3):
                  return f"📭 Mock search results for: {query}\n(Azure Search tool not configured)"
            agents["search_tool"] = MockSearchTool()

      # Freshdesk integration removed - focusing on grounded search responses only

      logging.info("✅ All agents and tools initialized")

      # Check if running interactively or in batch mode
      import sys
      if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
            await run_interactive_mode(agents)
      else:
            await run_batch_tests(agents)

   except Exception as e:
      logging.error(f"System initialization failed: {e}")
      print(f"❌ Failed to start system: {e}")

      # Try to run with minimal configuration
      logging.info("Attempting to run with minimal configuration...")
      try:
            minimal_agents = {
               "planner": await build_planner_agent(),
               "hr": await build_hr_agent(),
               "compliance": await build_compliance_agent(), 
               "finance": await build_finance_agent(),
               "search_tool": type('MockSearch', (), {'search': lambda self, q, top=3: f"Mock search for: {q}"})()
            }
            await run_batch_tests(minimal_agents)
      except Exception as minimal_error:
            print(f"❌ Even minimal configuration failed: {minimal_error}")

if __name__ == "__main__":
   asyncio.run(main())
