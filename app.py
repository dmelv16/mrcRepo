# app.py - Updated Streamlit Dashboard for Artemis Predictions with Table View Priority

import streamlit as st
import pandas as pd
import json
from datetime import datetime
import os
import plotly.express as px
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="MLB AI Betting Picks - Artemis",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    /* Fix metric containers */
    [data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1) !important;
        border: 1px solid rgba(28, 131, 225, 0.2) !important;
        padding: 15px !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
    }
    
    /* Ensure metric text is visible */
    [data-testid="metric-container"] > div {
        color: #262730 !important;
    }
    
    /* Style metric labels */
    [data-testid="metric-container"] label {
        color: #555 !important;
        font-weight: 600 !important;
    }
    
    /* Style metric values */
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #262730 !important;
        font-weight: 700 !important;
    }
    
    /* Style delta values */
    [data-testid="metric-container"] [data-testid="metric-delta-icon-container"] {
        color: #00cc44 !important;
    }
    
    /* Enhanced table styling */
    .stDataFrame {
        font-size: 14px !important;
    }
    
    /* Style dataframe headers */
    .stDataFrame th {
        background-color: #1f77b4 !important;
        color: white !important;
        font-weight: bold !important;
        text-align: center !important;
        padding: 10px !important;
    }
    
    /* Style dataframe cells */
    .stDataFrame td {
        text-align: center !important;
        padding: 8px !important;
    }
    
    /* Bet card styling with better contrast */
    .bet-card {
        background-color: #f8f9fa !important;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border: 1px solid #e9ecef;
    }
    
    .bet-card h4 {
        color: #1a1a1a !important;
        margin-bottom: 10px;
    }
    
    .bet-card p {
        color: #333 !important;
        margin: 5px 0;
    }
    
    .bet-card strong {
        color: #000 !important;
    }
    
    /* Bet strength indicators */
    .strong-bet {
        border-left: 5px solid #00cc44;
        background-color: rgba(0, 204, 68, 0.05) !important;
    }
    
    .medium-bet {
        border-left: 5px solid #ffaa00;
        background-color: rgba(255, 170, 0, 0.05) !important;
    }
    
    .small-bet {
        border-left: 5px solid #6495ED;
        background-color: rgba(100, 149, 237, 0.05) !important;
    }
    
    /* Fix expander backgrounds */
    .streamlit-expanderHeader {
        background-color: rgba(28, 131, 225, 0.1) !important;
        border-radius: 5px;
    }
    
    /* Fix info/warning boxes */
    .stAlert {
        background-color: rgba(255, 255, 255, 0.95) !important;
        border: 1px solid rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Ensure tab content is readable */
    .stTabs [data-baseweb="tab-panel"] {
        background-color: transparent !important;
    }
    
    /* Fix column backgrounds */
    [data-testid="column"] {
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Data Loading Functions ---
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_daily_predictions():
    """Loads the latest daily predictions from the JSON file."""
    path = r"C:\Users\DMelv\Documents\bettingModelBaseball\apiBaseball\pipeline\artemis\json\daily_predictions.json"
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    return None

@st.cache_data
def load_backtest_history():
    """Loads the historical backtest report."""
    path = r"C:\Users\DMelv\Documents\bettingModelBaseball\apiBaseball\pipeline\artemis\json\backtest_history.json"
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    return None

# --- Helper Functions ---
def format_odds(decimal_odds):
    """Convert decimal odds to American format for display."""
    if decimal_odds >= 2.0:
        return f"+{int((decimal_odds - 1) * 100)}"
    else:
        return f"{int(-100 / (decimal_odds - 1))}"

def get_bet_color(strength):
    """Return color based on bet strength."""
    colors = {
        'STRONG': '#00cc44',
        'MEDIUM': '#ffaa00',
        'SMALL': '#6495ED'
    }
    return colors.get(strength, '#888888')

def style_dataframe(df):
    """Apply custom styling to the dataframe."""
    def highlight_strength(val):
        if val == 'STRONG':
            return 'background-color: rgba(0, 204, 68, 0.2); color: #00cc44; font-weight: bold'
        elif val == 'MEDIUM':
            return 'background-color: rgba(255, 170, 0, 0.2); color: #ff8800; font-weight: bold'
        elif val == 'SMALL':
            return 'background-color: rgba(100, 149, 237, 0.2); color: #4169E1; font-weight: bold'
        return ''
    
    def highlight_positive(val):
        try:
            num_val = float(str(val).replace('%', ''))
            if num_val > 0:
                return 'color: #00cc44; font-weight: bold'
            elif num_val < 0:
                return 'color: #ff4444'
        except:
            return ''
        return ''
    
    return df.style.applymap(highlight_strength, subset=['Strength']).applymap(
        highlight_positive, subset=['Edge %', 'EV %']
    )

# --- Helper function to display bets in card format ---
def display_bets_cards(bets):
    """Display bets in a card format."""
    for bet in bets:
        # Create columns for bet display
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Use Streamlit native components instead of HTML for better visibility
            st.markdown(f"### {bet['matchup']}")
            st.markdown(f"**Bet on:** {bet['bet_team']} @ {format_odds(bet['odds'])}")
            st.markdown(f"**Pitchers:** {bet['pitchers']['away']} vs {bet['pitchers']['home']}")
        
        with col2:
            st.metric("Stake", f"{bet['stake_pct']:.2f}%")
            st.metric("Model Edge", f"{bet['value_edge']:.1f}%")
        
        with col3:
            st.metric("Win Probability", f"{bet['model_win_prob']:.1f}%")
            st.metric("Expected Value", f"{bet['expected_value']:.1f}%")
        
        # Score predictions
        st.markdown(f"**Score Prediction:** {bet['away_score_pred']:.1f} - {bet['home_score_pred']:.1f} (Total: {bet['total_pred']:.1f})")
        
        # Add visual indicator for bet strength
        strength_emoji = {"STRONG": "🟢", "MEDIUM": "🟡", "SMALL": "🔵"}
        st.markdown(f"**Bet Strength:** {strength_emoji.get(bet['bet_strength'], '')} {bet['bet_strength']}")
        
        if bet.get('high_variance_flag'):
            st.warning("⚠️ High model variance - less reliable prediction")
        
        st.markdown("---")

def create_detailed_table(predictions):
    """Create a detailed table view of predictions."""
    df = pd.DataFrame(predictions)
    
    # Build display dataframe with available fields
    display_data = {
        'Matchup': df['matchup'],
        'Bet On': df['bet_team'],
        'Odds': df['odds'].apply(format_odds),
        'Stake %': df['stake_pct'].round(2),
        'Strength': df['bet_strength'],
        'Win Prob %': df['model_win_prob'].round(1),
    }
    
    # Add optional fields if they exist
    if 'market_win_prob' in df.columns:
        display_data['Market Prob %'] = df['market_win_prob'].round(1)
    
    display_data['Edge %'] = df['value_edge'].round(1)
    display_data['EV %'] = df['expected_value'].round(1)
    
    # Add pitcher information
    if 'pitchers' in df.columns:
        display_data['Away Pitcher'] = df['pitchers'].apply(lambda x: x.get('away', 'N/A'))
        display_data['Home Pitcher'] = df['pitchers'].apply(lambda x: x.get('home', 'N/A'))
    
    # Add score predictions if available
    if 'away_score_pred' in df.columns and 'home_score_pred' in df.columns:
        display_data['Score Pred'] = df.apply(lambda x: f"{x['away_score_pred']:.1f} - {x['home_score_pred']:.1f}", axis=1)
    
    if 'total_pred' in df.columns:
        display_data['Total Pred'] = df['total_pred'].round(1)
    
    # Add high variance flag
    if 'high_variance_flag' in df.columns:
        display_data['High Var'] = df['high_variance_flag'].apply(lambda x: '⚠️' if x else '')
    else:
        display_data['High Var'] = ''
    
    display_df = pd.DataFrame(display_data)
    
    return display_df

# --- Load Data ---
daily_data = load_daily_predictions()
history_data = load_backtest_history()

# --- Sidebar ---
with st.sidebar:
    st.header("⚾ MLB AI Betting System")
    st.markdown("### Artemis Predictor v1.0")
    
    if daily_data:
        st.info(f"📅 Date: {daily_data['date']}")
        st.info(f"🔄 Updated: {daily_data['last_updated']}")
    
    st.markdown("---")
    st.markdown("""
    ### Bet Strength Guide
    - 🟢 **STRONG**: 4%+ of bankroll
    - 🟡 **MEDIUM**: 2.5-4% of bankroll
    - 🔵 **SMALL**: <2.5% of bankroll
    """)
    
    if daily_data and 'model_info' in daily_data:
        st.markdown("---")
        st.markdown("### Model Information")
        st.metric("Models Used", daily_data['model_info']['models_used'])
        st.metric("Profit Threshold", f"{daily_data['model_info']['threshold']:.3f}")

# --- Main Dashboard ---
st.title("⚾ MLB AI Betting Predictions - Artemis System")

# --- Historical Performance Section ---
if history_data:
    st.header("📊 Historical Performance")
    
    summary = history_data.get('backtest_summary', {})
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("ROI", f"{summary.get('roi', 0):.2f}%", 
                help="Return on Investment across all bets")
    col2.metric("Total P&L", f"${summary.get('total_pnl', 0):,.2f}",
                help="Total profit/loss in dollars")
    col3.metric("Win Rate", f"{summary.get('win_rate', 0) * 100:.1f}%",
                help="Percentage of winning bets")
    col4.metric("Total Bets", f"{summary.get('total_bets', 0):,}",
                help="Total number of bets placed")
    col5.metric("Avg Bet Size", f"{summary.get('avg_bet_size', 2.5):.1f}%",
                help="Average Kelly bet size")

    # Historical chart
    if 'detailed_bets' in history_data:
        history_df = pd.DataFrame(history_data['detailed_bets'])
        if not history_df.empty and 'date' in history_df.columns and 'bankroll_after' in history_df.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(history_df['date']),
                y=history_df['bankroll_after'],
                mode='lines',
                name='Bankroll',
                line=dict(color='#00cc44', width=2)
            ))
            fig.update_layout(
                title="Bankroll Growth Over Time",
                xaxis_title="Date",
                yaxis_title="Bankroll ($)",
                hovermode='x unified',
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# --- Today's Predictions Section ---
st.header("🎯 Today's Recommended Bets")

if daily_data and 'predictions' in daily_data:
    # Summary metrics
    summary = daily_data['summary']
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Games", summary['total_games'])
    col2.metric("Recommended Bets", summary['total_bets'])
    col3.metric("Total Stake", f"{summary['total_stake_pct']:.1f}%")
    col4.metric("Expected Return", f"{summary['expected_return_pct']:.1f}%",
                delta=f"{summary['expected_return_pct']:.1f}%")
    
    # Bet strength breakdown
    st.subheader("Bet Distribution")
    strength_data = summary['bets_by_strength']
    
    col1, col2, col3 = st.columns(3)
    col1.metric("🟢 Strong Bets", strength_data['strong'])
    col2.metric("🟡 Medium Bets", strength_data['medium'])
    col3.metric("🔵 Small Bets", strength_data['small'])
    
    # Display bets - TABLE VIEW AS PRIMARY
    st.subheader("Detailed Bet Recommendations")
    
    predictions = daily_data['predictions']
    
    if predictions:
        # Create tabs with table view first
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 All Bets (Table)", "🟢 Strong", "🟡 Medium", "🔵 Small", "📋 Card View"])
        
        with tab1:
            st.markdown("### Complete Betting Table")
            display_df = create_detailed_table(predictions)
            
            # Use Streamlit's native dataframe with column configuration
            st.dataframe(
                display_df,
                column_config={
                    "Stake %": st.column_config.ProgressColumn(
                        "Stake %", format="%.2f%%", min_value=0, max_value=5
                    ),
                    "Win Prob %": st.column_config.ProgressColumn(
                        "Win Prob %", format="%.1f%%", min_value=0, max_value=100
                    ),
                    "Market Prob %": st.column_config.NumberColumn(
                        "Market Prob %", format="%.1f%%"
                    ),
                    "Edge %": st.column_config.NumberColumn(
                        "Edge %", format="%.1f%%"
                    ),
                    "EV %": st.column_config.NumberColumn(
                        "EV %", format="%.1f%%"
                    ),
                    "Total Pred": st.column_config.NumberColumn(
                        "Total Pred", format="%.1f"
                    ),
                    "Strength": st.column_config.TextColumn(
                        "Strength",
                        help="Bet strength based on Kelly criterion"
                    ),
                    "High Var": st.column_config.TextColumn(
                        "⚠️", 
                        help="High variance flag"
                    )
                },
                hide_index=True,
                use_container_width=True,
                height=400
            )
            
            # Download button for CSV
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"artemis_bets_{daily_data['date']}.csv",
                mime="text/csv",
            )
        
        with tab2:
            strong_bets = [b for b in predictions if b['bet_strength'] == 'STRONG']
            if strong_bets:
                st.markdown("### Strong Bets (4%+ Kelly)")
                strong_df = create_detailed_table(strong_bets)
                st.dataframe(
                    strong_df,
                    column_config={
                        "Stake %": st.column_config.ProgressColumn(
                            "Stake %", format="%.2f%%", min_value=0, max_value=5
                        ),
                        "Win Prob %": st.column_config.ProgressColumn(
                            "Win Prob %", format="%.1f%%", min_value=0, max_value=100
                        ),
                    },
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.info("No strong bets for today.")
        
        with tab3:
            medium_bets = [b for b in predictions if b['bet_strength'] == 'MEDIUM']
            if medium_bets:
                st.markdown("### Medium Bets (2.5-4% Kelly)")
                medium_df = create_detailed_table(medium_bets)
                st.dataframe(
                    medium_df,
                    column_config={
                        "Stake %": st.column_config.ProgressColumn(
                            "Stake %", format="%.2f%%", min_value=0, max_value=5
                        ),
                        "Win Prob %": st.column_config.ProgressColumn(
                            "Win Prob %", format="%.1f%%", min_value=0, max_value=100
                        ),
                    },
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.info("No medium bets for today.")
        
        with tab4:
            small_bets = [b for b in predictions if b['bet_strength'] == 'SMALL']
            if small_bets:
                st.markdown("### Small Bets (<2.5% Kelly)")
                small_df = create_detailed_table(small_bets)
                st.dataframe(
                    small_df,
                    column_config={
                        "Stake %": st.column_config.ProgressColumn(
                            "Stake %", format="%.2f%%", min_value=0, max_value=5
                        ),
                        "Win Prob %": st.column_config.ProgressColumn(
                            "Win Prob %", format="%.1f%%", min_value=0, max_value=100
                        ),
                    },
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.info("No small bets for today.")
        
        with tab5:
            st.markdown("### Card View (Alternative Display)")
            display_bets_cards(predictions)
        
        # Quick summary statistics
        st.markdown("---")
        st.subheader("📈 Quick Statistics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_edge = sum(b['value_edge'] for b in predictions) / len(predictions)
            st.metric("Average Edge", f"{avg_edge:.1f}%")
        
        with col2:
            avg_ev = sum(b['expected_value'] for b in predictions) / len(predictions)
            st.metric("Average EV", f"{avg_ev:.1f}%")
        
        with col3:
            high_var_count = sum(1 for b in predictions if b.get('high_variance_flag', False))
            st.metric("High Variance Bets", high_var_count)
        
    else:
        st.info("No bets recommended for today.")
        
else:
    st.warning("⚠️ Today's predictions have not been generated yet. Please run the artemisPredictor.py script.")

# --- Risk Management Section ---
if daily_data and 'predictions' in daily_data and daily_data['predictions']:
    st.markdown("---")
    st.header("⚠️ Risk Management")
    
    total_stake = daily_data['summary']['total_stake_pct']
    
    if total_stake > 15:
        st.warning(f"⚠️ Large total allocation ({total_stake:.1f}%) - consider scaling down bet sizes")
    
    # Check for high variance bets
    high_var_bets = [b for b in daily_data['predictions'] if b.get('high_variance_flag', False)]
    if high_var_bets:
        st.warning(f"⚠️ {len(high_var_bets)} bet(s) have high model variance - these predictions are less reliable")
    
    # Kelly criterion explanation
    with st.expander("ℹ️ Understanding Kelly Criterion"):
        st.markdown("""
        The betting sizes are calculated using the Kelly Criterion, which determines optimal bet sizing based on:
        - **Edge**: How much better our predicted probability is than the market's implied probability
        - **Odds**: The potential payout of the bet
        
        We use a conservative 25% Kelly multiplier for safety. This means:
        - If full Kelly suggests betting 20% of bankroll, we bet 5%
        - Maximum bet size is capped at 5% per game
        - This protects against model uncertainty and variance
        """)

# --- Footer ---
st.markdown("---")
st.caption("Data updates every time artemisPredictor.py is run. Bet responsibly.")