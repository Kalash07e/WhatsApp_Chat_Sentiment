# MEDIUM PRIORITY IMPROVEMENTS

#### 4. **Performance Optimizations**

# SUGGESTION 4: Caching for Better Performance
@st.cache_data
def process_large_chat_files(uploaded_file):
    """Cache processed data to avoid recomputation"""
    try:
        # Read file efficiently
        if uploaded_file.type == "text/plain":
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8", errors='ignore'))
            data = stringio.read()
        else:
            st.error("Unsupported file type")
            return None
        
        # Process in chunks for large files
        lines = data.split('\n')
        chunk_size = 1000  # Process 1000 lines at a time
        processed_chunks = []
        
        for i in range(0, len(lines), chunk_size):
            chunk = lines[i:i+chunk_size]
            chunk_df = preprocess_chat(chunk)
            if chunk_df is not None and not chunk_df.empty:
                processed_chunks.append(chunk_df)
        
        if processed_chunks:
            return pd.concat(processed_chunks, ignore_index=True)
        return None
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

# SUGGESTION 5: Smart Data Sampling for Large Datasets
def smart_sample_data(df, max_messages=5000):
    """Intelligently sample data while preserving key insights"""
    if len(df) <= max_messages:
        return df
    
    # Preserve recent messages (more relevant)
    recent_count = int(max_messages * 0.4)  # 40% recent
    random_count = max_messages - recent_count  # 60% random sample
    
    recent_df = df.tail(recent_count)
    remaining_df = df.iloc[:-recent_count]
    
    if len(remaining_df) > 0:
        sampled_df = remaining_df.sample(n=min(random_count, len(remaining_df)))
        return pd.concat([sampled_df, recent_df]).sort_values('datetime_parsed')
    
    return recent_df

# SUGGESTION 6: Multi-threading for Analysis
import concurrent.futures
import threading

def parallel_sentiment_analysis(messages, batch_size=100):
    """Process sentiment analysis in parallel batches"""
    def analyze_batch(batch):
        return [TextBlob(msg).sentiment.polarity for msg in batch]
    
    # Split messages into batches
    batches = [messages[i:i+batch_size] for i in range(0, len(messages), batch_size)]
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_batch = {executor.submit(analyze_batch, batch): batch for batch in batches}
        
        for future in concurrent.futures.as_completed(future_to_batch):
            try:
                batch_results = future.result()
                results.extend(batch_results)
            except Exception as e:
                st.warning(f"Error in batch processing: {e}")
                # Fallback to single-threaded
                batch = future_to_batch[future]
                results.extend([TextBlob(msg).sentiment.polarity for msg in batch])
    
    return results

#### 5. **Advanced UI/UX Enhancements**

# SUGGESTION 7: Dark/Light Theme Toggle
def apply_custom_theme():
    """Apply custom theme with toggle option"""
    theme = st.sidebar.selectbox("Choose Theme", ["Dark", "Light", "Auto"])
    
    if theme == "Dark":
        st.markdown("""
        <style>
        .main-header { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem; 
            border-radius: 15px; 
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }
        .metric-card { 
            background: linear-gradient(145deg, #2d3748, #4a5568);
            padding: 1.5rem; 
            border-radius: 10px; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            border: 1px solid #4a5568;
        }
        .insight-box { 
            background: rgba(26, 32, 44, 0.8); 
            border-left: 4px solid #58a6ff; 
            padding: 1rem; 
            margin: 1rem 0;
            border-radius: 5px;
        }
        </style>
        """, unsafe_allow_html=True)
    elif theme == "Light":
        st.markdown("""
        <style>
        .main-header { 
            background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
            color: white;
        }
        .metric-card { 
            background: linear-gradient(145deg, #f8f9fa, #e9ecef);
            border: 1px solid #dee2e6;
        }
        </style>
        """, unsafe_allow_html=True)

# SUGGESTION 8: Interactive Filtering System
def create_advanced_filters(df):
    """Create comprehensive filtering system"""
    st.sidebar.header("🔍 Advanced Filters")
    
    # Date range filter
    if 'datetime_parsed' in df.columns:
        min_date = df['datetime_parsed'].min().date()
        max_date = df['datetime_parsed'].max().date()
        
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            df = df[
                (df['datetime_parsed'].dt.date >= date_range[0]) & 
                (df['datetime_parsed'].dt.date <= date_range[1])
            ]
    
    # User filter
    authors = df['author'].unique().tolist()
    if 'System' in authors:
        authors.remove('System')
    
    selected_authors = st.sidebar.multiselect(
        "Select Users",
        options=authors,
        default=authors
    )
    
    if selected_authors:
        df = df[df['author'].isin(selected_authors)]
    
    # Sentiment filter
    sentiment_filter = st.sidebar.select_slider(
        "Sentiment Range",
        options=["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"],
        value=("Very Negative", "Very Positive")
    )
    
    # Message length filter
    if 'message' in df.columns:
        df['message_length'] = df['message'].str.len()
        min_length, max_length = st.sidebar.slider(
            "Message Length Range",
            min_value=0,
            max_value=int(df['message_length'].max()),
            value=(0, int(df['message_length'].max())),
            step=10
        )
        
        df = df[
            (df['message_length'] >= min_length) & 
            (df['message_length'] <= max_length)
        ]
    
    return df

# SUGGESTION 9: Export Options
def create_export_options(df, analysis_results):
    """Add comprehensive export functionality"""
    st.sidebar.header("📁 Export Options")
    
    # Export raw data
    if st.sidebar.button("📊 Export Data as CSV"):
        csv = df.to_csv(index=False)
        st.sidebar.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"whatsapp_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Export analysis summary
    if st.sidebar.button("📋 Export Analysis Summary"):
        summary = create_analysis_summary(df, analysis_results)
        st.sidebar.download_button(
            label="Download Summary",
            data=summary,
            file_name=f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    # Export visualizations
    if st.sidebar.button("🎨 Export Charts as Images"):
        # This would require additional implementation
        st.sidebar.info("Chart export functionality coming soon!")

def create_analysis_summary(df, analysis_results):
    """Create a text summary of analysis"""
    summary = f"""
WhatsApp Chat Analysis Summary
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Chat Overview:
- Total Messages: {len(df)}
- Unique Users: {len(df['author'].unique()) - (1 if 'System' in df['author'].unique() else 0)}
- Date Range: {df['datetime_parsed'].min()} to {df['datetime_parsed'].max()}
- Duration: {(df['datetime_parsed'].max() - df['datetime_parsed'].min()).days} days

Sentiment Analysis:
- Average Sentiment: {df['sentiment'].mean():.3f}
- Most Positive User: {df.groupby('author')['sentiment'].mean().idxmax()}
- Most Active User: {df['author'].value_counts().index[0]}

Key Insights:
{analysis_results.get('insights', 'No specific insights generated')}
    """
    return summary
