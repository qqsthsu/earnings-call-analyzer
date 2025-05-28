import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import sent_tokenize
import re
import spacy
from collections import Counter
from wordcloud import WordCloud

# Make sure to download necessary NLTK resources
try:
    nltk.download('punkt')
    nltk.download('stopwords')
except:
    pass

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def read_earnings_call_transcript(file_path):
    with open(file_path, 'r', encoding='ISO-8859-1') as file:
        content = file.read()
    return content

def extract_sections(content):
    """Extract key sections from transcript"""
    sections = {}
    
    # Extract prepared remarks
    prepared_match = re.search(r'I will now turn the conference over to.*?(.*?)Question-and-Answer Session', 
                              content, re.DOTALL)
    if prepared_match:
        sections['prepared_remarks'] = prepared_match.group(1)
    else:
        sections['prepared_remarks'] = ""
    
    # Extract Q&A
    qa_match = re.search(r'Question-and-Answer Session(.*?)This does conclude', content, re.DOTALL)
    if qa_match:
        sections['qa'] = qa_match.group(1)
    else:
        sections['qa'] = ""
    
    return sections

def extract_executive_statements(content):
    """Extract statements from key executives"""
    executives = {
        'Andrew Witty': [], # CEO
        'John Rex': [],     # CFO
        'Dirk McMahon': []  # President/COO
    }
    
    for exec_name in executives.keys():
        pattern = re.compile(f'{exec_name}(.*?)(?:(?:{"|".join(executives.keys())})|Operator|Conference Call Participants)',
                           re.IGNORECASE | re.DOTALL)
        matches = pattern.findall(content)
        if matches:
            for match in matches:
                executives[exec_name].append(match.strip())
    
    return executives

def analyze_strategic_focus(transcripts):
    """Identify strategic focus areas across calls"""
    # Key strategic areas to track
    strategic_areas = [
        'value-based care', 'medicare advantage', 'medicare', 'medicaid', 'optum health', 
        'optum insight', 'optum rx', 'pharmacy', 'growth', 'digital', 'technology', 
        'artificial intelligence', 'ai', 'innovation', 'expense', 'cost', 'margin',
        'engagement', 'behavioral health', 'home health', 'revenue', 'care delivery',
        'telehealth', 'virtual care', 'consumer experience', 'value proposition'
    ]
    
    results = {}
    for file_name, content in transcripts.items():
        # Get quarter from filename
        quarter_match = re.search(r'(\d{4}\s*Q[1-4])', file_name)
        quarter = quarter_match.group(1) if quarter_match else file_name
        
        # Count mentions of each area
        counts = {}
        for area in strategic_areas:
            counts[area] = len(re.findall(r'\b' + re.escape(area) + r'\b', content.lower()))
        
        results[quarter] = counts
    
    # Convert to DataFrame
    df = pd.DataFrame(results).T
    
    return df

def extract_competitive_intelligence(transcripts):
    """Extract competitive intelligence including products, strategies, and market positioning"""
    
    # Key product/service terms to track
    product_terms = [
        'optum care', 'optum health', 'optum insight', 'optum rx', 'optum financial',
        'medicare advantage', 'medicare', 'medicaid', 'commercial', 'employer', 'individual',
        'dual eligible', 'value-based care', 'pharmacy benefit', 'virtual care', 'telehealth',
        'behavioral health', 'home health', 'community care'
    ]
    
    # Key business strategy terms
    strategy_terms = [
        'growth', 'innovation', 'expansion', 'acquisition', 'partnership', 'investment',
        'digital transformation', 'artificial intelligence', 'ai', 'consumer experience',
        'cost management', 'efficiency', 'productivity', 'risk', 'value-based', 'engagement'
    ]
    
    # Competitive positioning terms
    competitive_terms = [
        'market share', 'competitive advantage', 'differentiator', 'leading', 'outperform',
        'margin', 'competitor', 'competition', 'challenge', 'opportunity', 'price',
        'quality', 'star rating', 'network', 'retention', 'satisfaction'
    ]
    
    # Track mentions across transcripts
    product_mentions = {}
    strategy_mentions = {}
    competitive_mentions = {}
    
    for file_name, content in transcripts.items():
        # Get quarter from filename
        quarter_match = re.search(r'(\d{4}\s*Q[1-4])', file_name)
        quarter = quarter_match.group(1) if quarter_match else file_name
        
        # Count product mentions
        product_counts = {}
        for term in product_terms:
            product_counts[term] = len(re.findall(r'\b' + re.escape(term) + r'\b', content.lower()))
        product_mentions[quarter] = product_counts
        
        # Count strategy mentions
        strategy_counts = {}
        for term in strategy_terms:
            strategy_counts[term] = len(re.findall(r'\b' + re.escape(term) + r'\b', content.lower()))
        strategy_mentions[quarter] = strategy_counts
        
        # Count competitive mentions
        comp_counts = {}
        for term in competitive_terms:
            comp_counts[term] = len(re.findall(r'\b' + re.escape(term) + r'\b', content.lower()))
        competitive_mentions[quarter] = comp_counts
    
    # Convert to DataFrames
    product_df = pd.DataFrame(product_mentions).T
    strategy_df = pd.DataFrame(strategy_mentions).T
    competitive_df = pd.DataFrame(competitive_mentions).T
    
    return product_df, strategy_df, competitive_df

def extract_competitor_mentions(transcripts):
    """Track mentions of competitors across earnings calls"""
    competitors = [
        'anthem', 'elevance', 'cigna', 'humana', 'cvs', 'aetna', 'centene', 
        'molina', 'blue cross', 'kaiser', 'oak street', 'cano health', 'village md',
        'one medical', 'clover health', 'bright health', 'alignment'
    ]
    
    results = {}
    for file_name, content in transcripts.items():
        # Get quarter from filename
        quarter_match = re.search(r'(\d{4}\s*Q[1-4])', file_name)
        quarter = quarter_match.group(1) if quarter_match else file_name
        
        # Count mentions
        counts = {}
        for competitor in competitors:
            counts[competitor] = len(re.findall(r'\b' + re.escape(competitor) + r'\b', content.lower()))
        
        results[quarter] = counts
    
    # Convert to DataFrame
    df = pd.DataFrame(results).T
    
    return df

def analyze_business_segments(transcripts):
    """Analyze performance and strategy by business segment"""
    segments = {
        'UnitedHealthcare': ['unitedhealth', 'uhc', 'health insurance', 'health benefit'],
        'Optum Health': ['optum health', 'optumhealth', 'care delivery', 'value-based'],
        'Optum Insight': ['optum insight', 'optuminsight', 'data analytics', 'revenue cycle'],
        'Optum Rx': ['optum rx', 'optumrx', 'pharmacy', 'prescription', 'drug']
    }
    
    segment_analysis = {}
    for file_name, content in transcripts.items():
        # Get quarter from filename
        quarter_match = re.search(r'(\d{4}\s*Q[1-4])', file_name)
        quarter = quarter_match.group(1) if quarter_match else file_name
        
        # Find sentences mentioning each segment
        sentences = sent_tokenize(content)
        segment_sentences = {segment: [] for segment in segments}
        
        for sentence in sentences:
            for segment, keywords in segments.items():
                if any(keyword.lower() in sentence.lower() for keyword in keywords):
                    segment_sentences[segment].append(sentence)
        
        segment_analysis[quarter] = segment_sentences
    
    # Analyze the segment data
    segment_data = {}
    for quarter, data in segment_analysis.items():
        quarter_data = {}
        for segment, sentences in data.items():
            # Count mentions
            quarter_data[f"{segment}_mentions"] = len(sentences)
            
            # Extract revenue information when available
            revenue_sentences = [s for s in sentences if "revenue" in s.lower() or "grew" in s.lower()]
            if revenue_sentences:
                quarter_data[f"{segment}_revenue_info"] = revenue_sentences[0]
            else:
                quarter_data[f"{segment}_revenue_info"] = ""
        
        segment_data[quarter] = quarter_data
    
    # Convert to DataFrame
    df = pd.DataFrame(segment_data).T
    
    return df, segment_analysis

def extract_growth_drivers(transcripts):
    """Identify key growth drivers mentioned in earnings calls"""
    growth_patterns = [
        r'growth (?:in|of|from|driven by) ([^.]+)',
        r'([^.]+) (?:drove|driving|contributed to|led to) (?:our|the) growth',
        r'growth strategy [^.]*includes ([^.]+)',
        r'key drivers? (?:of growth|for growth|of our growth) [^.]*(?:is|are|include) ([^.]+)'
    ]
    
    growth_drivers = {}
    for file_name, content in transcripts.items():
        # Get quarter from filename
        quarter_match = re.search(r'(\d{4}\s*Q[1-4])', file_name)
        quarter = quarter_match.group(1) if quarter_match else file_name
        
        # Extract growth drivers
        quarter_drivers = []
        for pattern in growth_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            quarter_drivers.extend(matches)
        
        growth_drivers[quarter] = quarter_drivers
    
    return growth_drivers

def create_visualizations(strategic_df, product_df, strategy_df, competitive_df, competitor_df):
    """Create visualizations for key insights"""
    # 1. Strategic focus areas word cloud
    plt.figure(figsize=(12, 8))
    strategic_sums = strategic_df.sum(axis=0)
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          colormap='YlGnBu').generate_from_frequencies(strategic_sums)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Strategic Focus Areas Word Cloud_Elevance', fontsize=16)
    plt.tight_layout()
    plt.savefig('Strategic Focus Areas Word Cloud_Elevance.png')
    
    # 2. Product mentions over time
    plt.figure(figsize=(14, 8))
    top_products = product_df.sum(axis=0).sort_values(ascending=False).head(8).index
    product_df[top_products].plot(kind='bar', figsize=(14, 8))
    plt.title('Top Product/Service Mentions by Quarter', fontsize=16)
    plt.ylabel('Number of Mentions', fontsize=12)
    plt.xlabel('Quarter', fontsize=12)
    plt.legend(title='Products/Services', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('product_mentions.png')
    
    # 3. Strategy word cloud
    plt.figure(figsize=(12, 8))
    strategy_sums = strategy_df.sum(axis=0)
    wordcloud = WordCloud(width=800, height=400, background_color='white', 
                         colormap='viridis').generate_from_frequencies(strategy_sums)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Business Strategy Focus', fontsize=16)
    plt.tight_layout()
    plt.savefig('strategy_wordcloud.png')
    
    # 4. Competitor mentions
    plt.figure(figsize=(12, 8))
    competitor_df.plot(kind='bar', stacked=True, figsize=(12, 8))
    plt.title('Competitor Mentions by Quarter', fontsize=16)
    plt.ylabel('Number of Mentions', fontsize=12)
    plt.xlabel('Quarter', fontsize=12)
    plt.legend(title='Competitors', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('competitor_mentions.png')
    
    # 5. Competitive positioning radar chart
    top_comp_terms = competitive_df.sum(axis=0).sort_values(ascending=False).head(8).index
    latest_quarter = competitive_df.index[-1]
    
    values = competitive_df.loc[latest_quarter, top_comp_terms].values
    categories = top_comp_terms.tolist()
    
    # Number of variables
    N = len(categories)
    
    # Create angles for each variable
    angles = [n / float(N) * 2 * 3.14159 for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Add the values for the last quarter
    values = list(values)
    values += values[:1]  # Close the loop
    
    # Create the plot
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    
    # Draw the lines and fill area
    plt.plot(angles, values, linewidth=1, linestyle='solid')
    plt.fill(angles, values, alpha=0.1)
    
    # Set labels
    plt.xticks(angles[:-1], categories, size=12)
    plt.yticks([])
    
    plt.title(f'Competitive Positioning Focus ({latest_quarter})', size=16)
    plt.tight_layout()
    plt.savefig('competitive_positioning.png')
    
    return

def main():
    # File paths
    import glob
    file_paths = glob.glob("/Users/user/Documents/Capstone/data/Elevance/*.txt")

    # Read all transcripts
    transcripts = {}
    for file_path in file_paths:
        transcripts[file_path] = read_earnings_call_transcript(file_path)
    
    # Analyze strategic focus
    strategic_df = analyze_strategic_focus(transcripts)
    print("Strategic focus areas analyzed")
    
    # Extract competitive intelligence
    product_df, strategy_df, competitive_df = extract_competitive_intelligence(transcripts)
    print("Competitive intelligence extracted")
    
    # Analyze competitor mentions
    competitor_df = extract_competitor_mentions(transcripts)
    print("Competitor mentions analyzed")
    
    # Analyze business segments
    segment_df, segment_analysis = analyze_business_segments(transcripts)
    print("Business segments analyzed")
    
    # Extract growth drivers
    growth_drivers = extract_growth_drivers(transcripts)
    print("Growth drivers extracted")
    
    # Create visualizations
    create_visualizations(strategic_df, product_df, strategy_df, competitive_df, competitor_df)
    print("Visualizations created")
    
    # Extract key strategic initiatives
    key_initiatives = {}
    for quarter, transcript in transcripts.items():
        sections = extract_sections(transcript)
        executive_statements = extract_executive_statements(transcript)
        
        # Focus on CEO statements for strategic initiatives
        ceo_text = ' '.join(executive_statements.get('Andrew Witty', []))
        
        # Process with spaCy for entity and phrase extraction
        doc = nlp(ceo_text)
        
        # Extract noun phrases as potential strategic initiatives
        noun_phrases = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]
        
        # Count frequency
        phrase_counter = Counter(noun_phrases)
        
        # Store top phrases
        key_initiatives[quarter] = phrase_counter.most_common(10)
    
    # Generate word cloud from all transcripts
    all_text = " ".join(transcripts.values())
    wordcloud = WordCloud(width=1200, height=600, background_color='white', colormap='tab10').generate(all_text)
    plt.figure(figsize=(14, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('transcripts_wordcloud.png')
    plt.close()
    print("Word cloud generated and saved as transcripts_wordcloud.png")

    # Generate comprehensive CI report
    report = {
        'strategic_focus': strategic_df,
        'product_mentions': product_df,
        'business_strategy': strategy_df,
        'competitive_positioning': competitive_df,
        'competitor_mentions': competitor_df,
        'business_segments': segment_df,
        'growth_drivers': growth_drivers,
        'key_initiatives': key_initiatives
    }
    
    # Print summary findings
    print("\n=== UnitedHealth Group Competitive Intelligence Analysis ===\n")
    
    # Top strategic focus areas
    print("Top Strategic Focus Areas:")
    top_strategic = strategic_df.sum().sort_values(ascending=False).head(10)
    for area, count in top_strategic.items():
        print(f"  - {area}: {count} mentions")
    
    # Top products/services
    print("\nTop Products/Services:")
    top_products = product_df.sum().sort_values(ascending=False).head(8)
    for product, count in top_products.items():
        print(f"  - {product}: {count} mentions")
    
    # Top business strategies
    print("\nTop Business Strategies:")
    top_strategies = strategy_df.sum().sort_values(ascending=False).head(8)
    for strategy, count in top_strategies.items():
        print(f"  - {strategy}: {count} mentions")
    
    # Competitor landscape
    print("\nCompetitor Landscape:")
    top_competitors = competitor_df.sum().sort_values(ascending=False).head(5)
    for competitor, count in top_competitors.items():
        print(f"  - {competitor}: {count} mentions")
    
    # Save results to CSV
    strategic_df.to_csv('strategic_focus.csv')
    product_df.to_csv('product_mentions.csv')
    strategy_df.to_csv('business_strategy.csv')
    competitive_df.to_csv('competitive_positioning.csv')
    competitor_df.to_csv('competitor_mentions.csv')
    segment_df.to_csv('business_segments.csv')
    
    print("\nAnalysis complete! Results saved to CSV files and visualizations generated.")
    
    return report

if __name__ == "__main__":
    main()