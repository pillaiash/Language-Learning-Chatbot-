# chatbot_ui.py

import gradio as gr
import random
import requests
from datetime import datetime
from textblob import TextBlob
import json
from config import DATABASE_PATH, OPENROUTER_API_KEY, CHALLENGE_DAILY_LIMIT, CHALLENGE_DIFFICULTY_LEVELS, SENTIMENT_THRESHOLD
import sqlite3  # Use SQLite instead of psycopg2

# Setup DB
conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)  # Connect to SQLite database
cursor = conn.cursor()

# Drop existing tables to ensure clean schema
cursor.execute("DROP TABLE IF EXISTS chats")
cursor.execute("DROP TABLE IF EXISTS mistakes")
cursor.execute("DROP TABLE IF EXISTS user_progress")

# Create enhanced tables for better tracking
cursor.execute("""
CREATE TABLE IF NOT EXISTS chats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_input TEXT,
    bot_response TEXT,
    sentiment_score REAL,
    scene TEXT,
    timestamp TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS mistakes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_input TEXT,
    mistake_type TEXT,
    correction TEXT,
    explanation TEXT,
    context TEXT,
    timestamp TEXT,
    review_count INTEGER DEFAULT 0,
    mastered BOOLEAN DEFAULT FALSE
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS user_progress (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_date TEXT,
    scene TEXT,
    total_interactions INTEGER,
    correct_responses INTEGER,
    mistakes_made INTEGER,
    confidence_score REAL
)
""")
conn.commit()

# OpenRouter API Setup
API_KEY = OPENROUTER_API_KEY
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}
MODEL = "mistralai/mistral-7b-instruct"

# Globals for Gradio
session_data = {
    "known_lang": "",
    "target_lang": "",
    "level": "",
    "scene": "",
    "chat_history": [],
    "current_challenge": None,
    "challenge_score": 0
}

# Scene options for each level
SCENE_OPTIONS = {
    "Beginner": ["ordering food at a restaurant", "greeting someone", "asking for directions"],
    "Intermediate": ["booking a hotel", "visiting a doctor", "chatting with a local"],
    "Advanced": ["debating social issues", "job interview", "discussing politics"]
}

def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def analyze_mistake(user_input, bot_response):
    """Analyze the type of mistake and provide structured feedback."""
    mistake_types = {
        "grammar": ["grammar", "tense", "conjugation", "structure", "incorrect"],
        "vocabulary": ["word choice", "meaning", "vocabulary", "wrong word"],
        "pronunciation": ["pronunciation", "accent", "sound", "pronounce"],
        "cultural": ["cultural context", "formal", "informal", "politeness"]
    }
    
    lower_response = bot_response.lower()
    for category, keywords in mistake_types.items():
        if any(keyword in lower_response for keyword in keywords):
            return category
    return "general"

def get_emotion_aware_response(sentiment_score, response):
    """Generate more natural, emotion-aware responses."""
    if sentiment_score < -0.3:
        encouragements = [
            "Don't worry! Everyone makes mistakes while learning. ",
            "That's a tricky one, but you're doing great! ",
            "Learning a language takes time, and you're making progress! ",
            "Keep going! Making mistakes is how we learn. "
        ]
        return f"{random.choice(encouragements)}{response} 💪"
    elif sentiment_score < 0:
        return f"Keep going! {response} You're getting better with every conversation! 🌟"
    elif sentiment_score > 0.3:
        celebrations = [
            "Fantastic! ",
            "That's perfect! ",
            "Excellent work! ",
            "You're really getting the hang of this! "
        ]
        return f"{random.choice(celebrations)}{response} 🎉"
    elif sentiment_score > 0.1:
        return f"Well done! {response} 👍"
    return response

def get_learning_insights():
    """Get enhanced learning insights from the database."""
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)  # Connect to SQLite database
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            mistake_type,
            COUNT(*) as count,
            SUM(CASE WHEN mastered THEN 1 ELSE 0 END) as mastered_count
        FROM mistakes 
        GROUP BY mistake_type 
        ORDER BY count DESC
    """)
    mistake_stats = cursor.fetchall()
    
    cursor.execute("""
        SELECT scene, COUNT(*) as count
        FROM chats
        GROUP BY scene
        ORDER BY count DESC
        LIMIT 3
    """)
    scene_stats = cursor.fetchall()
    
    insights = "📊 Learning Progress Report\n\n"
    
    if mistake_stats:
        insights += "🎯 Areas to Focus On:\n"
        for mistake_type, count, mastered in mistake_stats:
            progress = (mastered / count) * 100 if count > 0 else 0
            insights += f"- {mistake_type.title()}: {count} occurrences ({progress:.1f}% mastered)\n"
    
    if scene_stats:
        insights += "\n🗣️ Most Practiced Scenarios:\n"
        for scene, count in scene_stats:
            insights += f"- {scene}: {count} conversations\n"
    
    conn.close()  # Close the connection
    return insights

def query_openrouter(user_input, max_retries=3, retry_delay=1):
    """Query OpenRouter API with retry mechanism"""
    import time
    
    for attempt in range(max_retries):
        try:
            sentiment_score = analyze_sentiment(user_input)
            
            conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)  # Connect to SQLite database
            cursor = conn.cursor()
            
            # Get conversation context from recent history
            cursor.execute("""
                SELECT user_input, bot_response 
                FROM chats 
                ORDER BY timestamp DESC 
                LIMIT 3
            """)
            recent_context = cursor.fetchall()
            conn.close()  # Close the connection
            
            context_prompt = ""
            if recent_context:
                context_prompt = "Previous conversation:\n" + "\n".join([
                    f"User: {msg[0]}\nAssistant: {msg[1]}" 
                    for msg in recent_context[::-1]
                ]) + "\n\n"
            
            system_message = (
                f"You are Chatalyst, a friendly language guide. "
                f"Help the user learn {session_data['target_lang']} in a simple way. "
                f"Use short sentences and easy words. "
                f"Encourage them and provide examples. "
                f"Current scene: {session_data['scene']}.\n\n"
                f"{context_prompt}"
            )

            payload = {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_input}
                ],
                "temperature": 0.7,
                "max_tokens": 500
            }

            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=HEADERS,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                bot_response = response.json()["choices"][0]["message"]["content"]
                
                # Add motivational messages based on sentiment score
                if sentiment_score < -0.3:
                    bot_response = f"Don't worry! You're doing great! 😊 {bot_response}"
                elif sentiment_score > 0.3:
                    bot_response = f"Fantastic! Keep it up! 🎉 {bot_response}"
                
                return get_emotion_aware_response(sentiment_score, bot_response)
            else:
                print(f"OpenRouter API Error: {response.status_code} - {response.text}")
                return "I'm having trouble connecting right now. Could you please try again in a moment? 😊"

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return "The response is taking longer than expected. Could you please try again? 🕒"
        
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return "I'm having trouble connecting to my language services. Please check your internet connection and try again. 🌐"
        
        except Exception as e:
            print(f"Error in query_openrouter: {str(e)}")
            return "I seem to be having technical difficulties. Let's try that again! 🔄"
    
    return "I'm sorry, I'm having trouble responding right now. Please try again in a few moments. 🙏"

def chat(user_input, history):
    try:
        response = query_openrouter(user_input)
        sentiment_score = analyze_sentiment(user_input)
        
        # Initialize history if None
        history = history or []
        
        # Format messages according to Gradio's expected format
        user_message = {"role": "user", "content": user_input}
        assistant_message = {"role": "assistant", "content": response}
        
        # Add formatted messages to history
        history.append(user_message)
        history.append(assistant_message)
        
        # Store in database only if we got a valid response
        if not any(error_msg in response for error_msg in [
            "trouble connecting", 
            "taking longer than expected",
            "technical difficulties",
            "try again"
        ]):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)  # Connect to SQLite database
            cursor = conn.cursor()
            
            # Insert chat data
            cursor.execute("""
                INSERT INTO chats (user_input, bot_response, sentiment_score, scene, timestamp) 
                VALUES (?, ?, ?, ?, ?)
            """, (user_input, response, sentiment_score, session_data['scene'], timestamp))
            
            # Check for mistakes in the response
            mistake_type = analyze_mistake(user_input, response)
            if mistake_type != "general":  # Only store if a specific mistake type is found
                cursor.execute("""
                    INSERT INTO mistakes (user_input, mistake_type, correction, explanation, 
                    context, timestamp) VALUES (?, ?, ?, ?, ?, ?)
                """, (user_input, mistake_type, response, "Extracted from conversation", 
                      session_data['scene'], timestamp))
            
            conn.commit()  # Commit changes to the database
            conn.close()   # Close the connection
        
        return history, history
        
    except Exception as e:
        print(f"Error in chat function: {str(e)}")
        error_message = {"role": "assistant", "content": "I'm having trouble processing that. Let's try again! 🔄"}
        history = history or []
        history.append({"role": "user", "content": user_input})
        history.append(error_message)
        return history, history

def setup_user(known, target, level, scene):
    if not known or not target or not level or not scene:
        return "Please fill in all fields before starting the chat."
        
    session_data["known_lang"] = known
    session_data["target_lang"] = target
    session_data["level"] = level.capitalize()
    session_data["scene"] = scene

    greeting = f"Great! You're practicing {scene} in a {target}-speaking country. Let's begin!"
    return greeting

def update_scene_options(level):
    if not level:
        return gr.Dropdown(choices=[], label="Select a Scene")
    return gr.Dropdown(choices=SCENE_OPTIONS.get(level, SCENE_OPTIONS["Beginner"]), 
                      label="Select a Scene", 
                      value=SCENE_OPTIONS.get(level, SCENE_OPTIONS["Beginner"])[0])

def view_database_contents():
    """View contents of all tables in the database with improved formatting"""
    try:
        conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)  # Connect to SQLite database
        cursor = conn.cursor()
        
        # Get mistake statistics
        cursor.execute("""
            SELECT 
                mistake_type,
                COUNT(*) as count
            FROM mistakes 
            GROUP BY mistake_type 
            ORDER BY count DESC
        """)
        mistake_stats = cursor.fetchall()
        
        # Get recent mistakes
        cursor.execute("""
            SELECT user_input, correction, mistake_type, timestamp 
            FROM mistakes 
            ORDER BY timestamp DESC 
            LIMIT 5
        """)
        recent_mistakes = cursor.fetchall()
        
        output = """
        <div style="padding: 20px; background: white; border-radius: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
            <h3 style="color: #1f2937; font-size: 1.5rem; margin-bottom: 20px;">
                🎯 Your Learning Progress
            </h3>
            <div style="display: flex; flex-wrap: wrap; gap: 15px;">
        """
        
        # Add mistake type cards
        for mistake_type, count in mistake_stats:
            color = {
                "grammar": "#4f46e5",
                "vocabulary": "#059669",
                "pronunciation": "#db2777",
                "cultural": "#9333ea",
                "general": "#475569"
            }.get(mistake_type.lower(), "#475569")
            
            output += f"""
                <div style="background: {color}; color: white; padding: 15px; border-radius: 12px; flex: 1 1 150px; text-align: center;">
                    <h4 style="margin: 0;">{mistake_type.title()}</h4>
                    <p style="margin: 0;">{count} {'time' if count == 1 else 'times'}</p>
                </div>
            """
        
        output += """
            </div>
        </div>
        <div style="margin-top: 30px;">
            <h3 style="color: #1f2937; font-size: 1.5rem; margin-bottom: 20px;">
                💡 Recent Learning Opportunities
            </h3>
            <table style="width: 100%; border-collapse: collapse;">
                <thead>
                    <tr style="background: #f8fafc;">
                        <th style="padding: 10px; border: 1px solid #e5e7eb;">You Said</th>
                        <th style="padding: 10px; border: 1px solid #e5e7eb;">Correction</th>
                        <th style="padding: 10px; border: 1px solid #e5e7eb;">Mistake Type</th>
                        <th style="padding: 10px; border: 1px solid #e5e7eb;">Timestamp</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        # Check if there are recent mistakes
        if not recent_mistakes:
            output += """
                <tr>
                    <td colspan="4" style="padding: 10px; border: 1px solid #e5e7eb; text-align: center;">No recent mistakes recorded.</td>
                </tr>
            """
        else:
            # Add recent mistakes to the table
            for user_input, correction, mistake_type, timestamp in recent_mistakes:
                output += f"""
                    <tr>
                        <td style="padding: 10px; border: 1px solid #e5e7eb;">{user_input}</td>
                        <td style="padding: 10px; border: 1px solid #e5e7eb;">{correction}</td>
                        <td style="padding: 10px; border: 1px solid #e5e7eb;">{mistake_type.title()}</td>
                        <td style="padding: 10px; border: 1px solid #e5e7eb;">{timestamp}</td>
                    </tr>
                """
        
        output += """
                </tbody>
            </table>
        </div>
        </div>
        """
        
        conn.close()  # Close the connection
        return output
        
    except Exception as e:
        return f"""
        <div style="padding: 20px; background: white; border-radius: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
            <p style="color: #dc2626;">Unable to load learning progress. Please try again.</p>
        </div>
        """

# Update the Gradio interface layout
with gr.Blocks(
    title="Chatalyst - Your Language Learning Companion",
    theme=gr.themes.Soft(
        primary_hue="slate",
        secondary_hue="gray",
        neutral_hue="stone",
        font=["sans-serif", "ui-sans-serif", "system-ui"]
    ),
    css="container"
) as demo:
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            gr.Markdown(
                """
                <div style='text-align: center; padding: 2rem 0;'>
                    <h1 style='font-size: 2.5rem; margin-bottom: 1rem;'>🌍 Chatalyst</h1>
                    <p style='color: #94a3b8; font-size: 1.2rem;'>Your Smart Language Learning Companion</p>
                </div>
                """
            )
    
    with gr.Row(equal_height=True):
        # Setup Section in left column
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 🎯 Start Your Journey")
                with gr.Group():
                    known_input = gr.Textbox(
                        label="Your Known Language",
                        placeholder="e.g., English",
                        info="The language you're comfortable with"
                    )
                    target_input = gr.Textbox(
                        label="Target Language to Learn",
                        placeholder="e.g., Spanish",
                        info="The language you want to practice"
                    )
                    level_input = gr.Dropdown(
                        choices=["Beginner", "Intermediate", "Advanced"],
                        label="Your Level",
                        value="Beginner",
                        info="Select your proficiency level"
                    )
                    scene_input = gr.Dropdown(
                        choices=SCENE_OPTIONS["Beginner"],
                        label="Select a Scene",
                        info="Choose a conversation scenario"
                    )
                    start_btn = gr.Button(
                        "Begin Practice",
                        variant="primary",
                        size="lg"
                    )
                scene_output = gr.Markdown(
                    label="Session Info",
                    elem_classes=["session-info"]
                )
        
        # Chat Section in right column
        with gr.Column(scale=2):
            with gr.Group():
                gr.Markdown("### 💬 Practice Conversation")
                chatbot = gr.Chatbot(
                    height=600,
                    show_label=False,
                    bubble_full_width=False,
                    type="messages",
                    elem_classes=["chatbot"],
                    render=True,
                    avatar_images=["👤", "🤖"]
                )
                with gr.Row():
                    msg = gr.Textbox(
                        label="",
                        placeholder="Type your message here...",
                        show_label=False,
                        container=False,
                        scale=9,
                        autofocus=True
                    )
                    send_btn = gr.Button(
                        "Send",
                        variant="primary",
                        scale=1
                    )

    # Learning Insights and Database View in expandable sections
    with gr.Row():
        with gr.Column():
            with gr.Accordion("📊 Learning Insights", open=False):
                insights = gr.Markdown()
                refresh_insights = gr.Button("Refresh Insights")
            
            with gr.Accordion("🔍 Database Contents", open=False):
                db_viewer = gr.Markdown()
                refresh_db = gr.Button("Refresh Database View")

    state = gr.State([])

    # Custom CSS
    gr.Markdown("""
    <style>
        .gradio-container {
            max-width: 1400px !important;
            margin: 0 auto;
            padding: 2rem;
        }
        
        .gr-group {
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            background: #ffffff;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .session-info {
            padding: 1.5rem;
            background: #f8fafc;
            border-radius: 12px;
            margin: 1.5rem 0;
            color: #475569;
        }
        
        .chatbot {
            min-height: 600px;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            background: #f8fafc;
            margin: 1.5rem 0;
            padding: 1rem;
        }
        
        .gr-button-primary {
            background: #475569 !important;
            color: white !important;
            padding: 0.75rem 1.5rem !important;
            border-radius: 8px !important;
        }
        
        .gr-button-primary:hover {
            background: #334155 !important;
        }
        
        .gr-textbox, .gr-dropdown {
            background: #f8fafc !important;
            border-radius: 8px !important;
            padding: 0.75rem !important;
            margin: 0.5rem 0 !important;
        }
        
        h1, h2, h3 {
            color: #475569 !important;
            margin: 1.5rem 0 1rem 0 !important;
        }
        
        .gr-accordion {
            margin: 1.5rem 0 !important;
            border-radius: 12px !important;
        }
        
        .gr-accordion-header {
            padding: 1rem !important;
        }
        
        .database-content {
            padding: 1.5rem;
            background: #ffffff;
            border-radius: 12px;
            margin: 1rem 0;
            line-height: 1.6;
        }
        
        .database-section {
            margin: 1.5rem 0;
            padding: 1rem;
            background: #f8fafc;
            border-radius: 8px;
        }
        
        .database-entry {
            margin: 1rem 0;
            padding: 1rem;
            border-left: 4px solid #475569;
            background: #ffffff;
        }
        
        /* Add spacing between chat messages */
        .message {
            margin: 1rem 0 !important;
            padding: 1rem !important;
        }
    </style>
    """)

    # Event Handlers
    level_input.change(
        fn=update_scene_options,
        inputs=level_input,
        outputs=scene_input
    )

    start_btn.click(
        fn=setup_user,
        inputs=[known_input, target_input, level_input, scene_input],
        outputs=scene_output
    )

    send_btn.click(
        fn=chat,
        inputs=[msg, state],
        outputs=[chatbot, state]
    ).then(
        lambda: "", None, msg
    )

    refresh_insights.click(
        fn=get_learning_insights,
        outputs=insights
    )

    # Add event handler for database viewer
    refresh_db.click(
        fn=view_database_contents,
        outputs=db_viewer
    )

    # Update chat function to refresh database view after each message
    def chat_with_db_update(user_input, history):
        chat_result = chat(user_input, history)
        db_contents = view_database_contents()
        return chat_result[0], chat_result[1], db_contents

    # Update the send button click handler
    send_btn.click(
        fn=chat_with_db_update,
        inputs=[msg, state],
        outputs=[chatbot, state, db_viewer]
    ).then(
        lambda: "", None, msg
    )

# Launch UI
if __name__ == "__main__":
    try:
        print("Starting Chatalyst server...")
        demo.launch(
            server_name="127.0.0.1",  # Use localhost explicitly
            server_port=8080,         # Use port 8080 instead
            share=False,              # Don't create a public link
            show_error=True,          # Show detailed errors
            debug=True,               # Enable debug mode
            auth=None,                # No authentication required
            root_path="",             # No root path
            ssl_keyfile=None,         # No SSL
            ssl_certfile=None,        # No SSL
            ssl_keyfile_password=None, # No SSL password
            show_api=False            # Don't show API documentation
        )
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        print("Please check if port 8080 is available and not blocked by your firewall.") 
        