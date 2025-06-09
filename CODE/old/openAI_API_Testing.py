# # === 8. Use OpenAI Chat (Langchain Agent Equivalent) ===
# # def summarize_text(text):
# #     messages = [
# #         {"role": "system", "content": "You are a helpful assistant. Keep the answer short and suitable for Discord."},
# #         {"role": "user", "content": f"{text}"}
# #     ]
# #     response = openai.ChatCompletion.create(
# #         model="gpt-4o",
# #         messages=messages,
# #         max_tokens=300
# #     )
# #     return response['choices'][0]['message']['content']
# def check_vector_index(collection_name=MONGO_COLL, index_name=VECTOR_SEARCH):
#     try:
#         client = MongoClient(MONGO_URI)
#         db = client[MONGO_DB]
#         collection = db[collection_name]
#         # Try a test vector search to verify the index
#         pipeline = [
#             {
#                 '$vectorSearch': {
#                     'index': index_name,
#                     'queryVector': [0] * 768,  # Dummy vector of 768 dimensions
#                     'path': 'embedding',
#                     'numCandidates': 1,
#                     'limit': 0
#                 }
#             }
#         ]
#         list(collection.aggregate(pipeline))  # This will raise an error if the index doesn't exist
#         print(f"Vector search index '{index_name}' detected successfully.")
#         client.close()
#         return True
#     except Exception as e:
#         print(f"Error checking index: {e}")
#         client.close()
#         return False
    

    # 11. Discord bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Dictionary to track active tasks per user
active_tasks = {}

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}')
    if check_ollama_server():
        await initialize_document()
    else:
        print("Ollama server not available, skipping document initialization.")

@bot.command(name='search')
async def search_command(ctx, *, query):
    if ctx.channel.id != DISCORD_CHANNEL:
        return
    user_id = str(ctx.author.id)
    await ctx.send(f"Searching for: '{query}'...")
    
    if not DOCUMENT_READY:
        await ctx.send("Document not ready.")
        return

    try:
        # Create a task for the search
        task = asyncio.create_task(process_document(FILE_ID, query, user_id))
        active_tasks[user_id] = task
        similarities = await task
        
        if similarities and similarities[0]['similarity'] > 0.5:
            response = f"**Top Results for: '{query}'**\n\n"
            for i, result in enumerate(similarities[:3], 1):
                similarity_score = result['similarity']
                text_preview = result['text'][:500] + "..." if len(result['text']) > 500 else result['text']
                response += f"**{i}. Match (Similarity: {similarity_score:.3f})**\n{text_preview}\n\n"
        else:
            response = "No results found with sufficient similarity."
        
        if len(response) > 2000:
            response = response[:1997] + "..."
        
        await ctx.send(response)
        
    except asyncio.CancelledError:
        await ctx.send("Search operation cancelled.")
        return
    except Exception as e:
        await ctx.send(f"Error processing request: {str(e)}")
    finally:
        # Clean up task
        if user_id in active_tasks:
            del active_tasks[user_id]

@bot.command(name='chat')
async def chat_command(ctx, *, query):
    if ctx.channel.id != DISCORD_CHANNEL:
        return
    user_id = str(ctx.author.id)
    await ctx.send(f"Processing query: '{query}'...")
    
    if not DOCUMENT_READY:
        await ctx.send("Document not ready.")
        return

    try:
        # Create a task for document processing
        task = asyncio.create_task(process_document(FILE_ID, query, user_id))
        active_tasks[user_id] = task
        similarities = await task
        
        # Create a task for chat response
        task = asyncio.create_task(get_chat_response(query, similarities[:3], user_id))
        active_tasks[user_id] = task
        ollama_response = await task
        
        # Only send the Ollama response
        if len(ollama_response) > 2000:
            ollama_response = ollama_response[:1997] + "..."
        
        await ctx.send(ollama_response)
        
    except asyncio.CancelledError:
        await ctx.send("Chat operation cancelled.")
        return
    except Exception as e:
        await ctx.send(f"Error processing chat request: {str(e)}")
    finally:
        # Clean up task
        if user_id in active_tasks:
            del active_tasks[user_id]

@bot.command(name='info')
async def info_command(ctx):
    if ctx.channel.id != DISCORD_CHANNEL:
        return
    await ctx.send("**PDF Search Bot**\nSupports: PDF, Google Docs, Text\nCommands: `!search <query>`, `!chat <query>`, `!stop`, `!info`\nUsing 768D embeddings")

@bot.command(name='stop')
async def stop_command(ctx):
    if ctx.channel.id != DISCORD_CHANNEL:
        return
    user_id = str(ctx.author.id)
    
    # Check if the user has an active task
    if user_id in active_tasks:
        task = active_tasks[user_id]
        task.cancel()  # Attempt to cancel the task
        await ctx.send(f"Attempting to stop operation for {ctx.author.name}...")
    else:
        await ctx.send("No active operations to stop for you.")
    
    # Optionally, shut down the bot if the user is an admin
    if ctx.author.guild_permissions.administrator:
        await ctx.send("Shutting down the bot...")
        try:
            await bot.close()
            print(f"Bot stopped by {ctx.author.name}#{ctx.author.discriminator}")
        except Exception as e:
            await ctx.send(f"Error stopping the bot: {str(e)}")

# Main
if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)