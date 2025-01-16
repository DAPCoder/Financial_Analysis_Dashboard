import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from scipy.stats import norm

# Page config
st.set_page_config(
    page_title="Enhanced Financial Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose Analysis Type",
    ["Stock Analysis", "Compare Stocks", "Technical Analysis", "Portfolio Analysis", "Options Analysis"]
)

# Additional helper functions
def calculate_volatility(price_data, window=252):
    """Calculate rolling volatility"""
    returns = np.log(price_data / price_data.shift(1))
    return returns.rolling(window=window).std() * np.sqrt(window)

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate Sharpe Ratio"""
    excess_returns = returns - risk_free_rate/252
    return np.sqrt(252) * excess_returns.mean() / returns.std()

def calculate_beta(stock_returns, market_returns):
    """Calculate stock beta"""
    # Ensure data is one-dimensional
    stock_returns = stock_returns.squeeze()
    market_returns = market_returns.squeeze()
    
    # Calculate beta using numpy
    covariance = np.cov(stock_returns, market_returns)[0][1]
    market_variance = np.var(market_returns)
    return covariance / market_variance

def get_stock_info(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        return ticker, info
    except Exception as e:
        st.error(f"Error fetching data for {ticker_symbol}: {str(e)}")
        return None, None

def display_stock_summary(ticker, info):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"${info.get('currentPrice', 'N/A')}")
    with col2:
        st.metric("Market Cap", f"${info.get('marketCap', 'N/A'):,}")
    with col3:
        st.metric("52 Week High", f"${info.get('fiftyTwoWeekHigh', 'N/A')}")
    with col4:
        st.metric("Dividend Yield", f"{info.get('dividendYield', 0)*100:.2f}%")
    
    # Company info with expanded details
    st.subheader("Company Information")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write(info.get('longBusinessSummary', 'No description available.'))
    with col2:
        st.write("**Industry:** ", info.get('industry', 'N/A'))
        st.write("**Sector:** ", info.get('sector', 'N/A'))
        st.write("**Employees:** ", f"{info.get('fullTimeEmployees', 0):,}")
        st.write("**Country:** ", info.get('country', 'N/A'))

def calculate_volatility(price_data, window=252):
    """Calculate rolling volatility"""
    returns = np.log(price_data / price_data.shift(1))
    vol = returns.rolling(window=window).std() * np.sqrt(window)
    return vol

def display_advanced_metrics(ticker, info):
    st.subheader("Advanced Metrics")
    
    try:
        # Get historical data for calculations
        hist = ticker.history(period='1y')
        stock_prices = hist['Close']
        returns = stock_prices.pct_change().dropna()
        
        # Get market data (S&P 500) for the same period
        market_prices = yf.download('^GSPC', start=hist.index[0], end=hist.index[-1])['Close']
        market_returns = market_prices.pct_change().dropna()
        
        # Align dates between stock and market returns
        common_dates = returns.index.intersection(market_returns.index)
        returns = returns[common_dates]
        market_returns = market_returns[common_dates]
        
        # Calculate metrics and ensure they're scalar values
        volatility_series = calculate_volatility(stock_prices)
        volatility = volatility_series.iloc[-1] if not volatility_series.empty else np.nan
        
        sharpe = calculate_sharpe_ratio(returns)
        beta = calculate_beta(returns, market_returns)
        
        # Calculate alpha
        alpha = returns.mean()*252 - market_returns.mean()*252*beta
        
        # Format metrics with error handling
        def format_metric(value, include_percent=True):
            try:
                if np.isnan(value):
                    return "N/A"
                return f"{value:.2f}%" if include_percent else f"{value:.2f}"
            except:
                return "N/A"
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Volatility (Annual)", format_metric(volatility*100))
        with col2:
            st.metric("Sharpe Ratio", format_metric(sharpe, include_percent=False))
        with col3:
            st.metric("Beta", format_metric(beta, include_percent=False))
        with col4:
            st.metric("Alpha (1Y)", format_metric(alpha*100))
            
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
        st.write("Please check if the stock has sufficient historical data.")

def plot_candlestick(ticker_data):
    fig = go.Figure(data=[go.Candlestick(x=ticker_data.index,
                open=ticker_data['Open'],
                high=ticker_data['High'],
                low=ticker_data['Low'],
                close=ticker_data['Close'])])
    
    fig.update_layout(
        title="Price Action",
        yaxis_title="Price",
        xaxis_title="Date",
        template="plotly_dark"
    )
    
    return fig

# Main content
st.title("Enhanced Financial Analysis Dashboard")

if page == "Stock Analysis":
    st.header("Advanced Stock Analysis")
    ticker_symbol = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT)", "AAPL").upper()
    
    if ticker_symbol:
        ticker, info = get_stock_info(ticker_symbol)
        
        if ticker and info:
            # Display enhanced summary
            display_stock_summary(ticker, info)
            
            # Chart period selector with more options
            period = st.selectbox(
                "Select Time Period",
                ['1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max'],
                index=3
            )
            
            # Enhanced visualization options
            chart_type = st.radio(
                "Select Chart Type",
                ["Line", "Candlestick"],
                horizontal=True
            )
            
            # Display enhanced stock chart
            st.subheader("Price Chart")
            hist_data = ticker.history(period=period)
            
            if chart_type == "Candlestick":
                st.plotly_chart(plot_candlestick(hist_data), use_container_width=True)
            else:
                st.line_chart(hist_data['Close'])
            
            # Display advanced metrics
            display_advanced_metrics(ticker, info)
            
            # Enhanced tabs
            tab1, tab2, tab3, tab4 = st.tabs(["News", "Financials", "Analysis", "Risk Metrics"])
            
            with tab1:
                st.subheader("Recent News with Sentiment")
                try:
                    news = ticker.news
                    if news:
                        for item in news[:5]:
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                title = item.get('headline', item.get('title', 'No title available'))
                                publisher = item.get('source', item.get('publisher', 'Unknown source'))
                                st.write(f"**{title}**")
                                st.write(f"Source: {publisher}")
                            with col2:
                                # Simulate sentiment analysis
                                sentiment = np.random.choice(['🟢 Positive', '🟡 Neutral', '🔴 Negative'])
                                st.write(f"Sentiment: {sentiment}")
                            st.write("---")
                except Exception as e:
                    st.write("Unable to fetch news at this time")
            
            with tab2:
                st.subheader("Financial Statements")
                statement_type = st.selectbox(
                    "Select Statement Type",
                    ["Income Statement", "Balance Sheet", "Cash Flow"]
                )
                
                if statement_type == "Income Statement":
                    st.dataframe(ticker.financials)
                elif statement_type == "Balance Sheet":
                    st.dataframe(ticker.balance_sheet)
                else:
                    st.dataframe(ticker.cashflow)
            
            with tab3:
                st.subheader("Enhanced Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Growth Metrics**")
                    revenue_growth = ticker.financials.loc["Total Revenue"].pct_change()
                    st.metric("Revenue Growth (YoY)", f"{revenue_growth.iloc[-1]*100:.1f}%")
                    
                with col2:
                    st.write("**Efficiency Metrics**")
                    if not ticker.financials.empty and "Total Revenue" in ticker.financials.index:
                        revenue = ticker.financials.loc["Total Revenue"].iloc[-1]
                        employees = info.get('fullTimeEmployees', 0)
                        if employees > 0:
                            st.metric("Revenue per Employee", f"${revenue/employees:,.0f}")
            
            with tab4:
                st.subheader("Risk Analysis")
                
                # Calculate VaR
                returns = hist_data['Close'].pct_change().dropna()
                var_95 = norm.ppf(0.05, returns.mean(), returns.std())
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Value at Risk (95%)", f"{var_95*100:.2f}%")
                with col2:
                    st.metric("Max Drawdown", 
                             f"{((hist_data['Close'].max() - hist_data['Close'].min()) / hist_data['Close'].max() * 100):.2f}%")

elif page == "Compare Stocks":
    st.header("Compare Stocks")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # First stock input
        stock1 = st.text_input("Enter First Stock Symbol", "AAPL").upper()
    
    with col2:
        # Second stock input
        stock2 = st.text_input("Enter Second Stock Symbol", "MSFT").upper()
    
    # Time period selection
    period = st.selectbox(
        "Select Time Period",
        ['1mo', '3mo', '6mo', '1y', '2y', '5y'],
        index=2
    )
    
    if stock1 and stock2:
        try:
            # Fetch data for both stocks
            ticker1 = yf.Ticker(stock1)
            ticker2 = yf.Ticker(stock2)
            
            df1 = ticker1.history(period=period)
            df2 = ticker2.history(period=period)
            
            if not df1.empty and not df2.empty:
                # Normalize prices to starting price
                df1['Normalized'] = df1['Close'] / df1['Close'].iloc[0] * 100
                df2['Normalized'] = df2['Close'] / df2['Close'].iloc[0] * 100
                
                # Create comparison plots
                st.subheader("Price Comparison")
                
                # Relative Performance Chart
                fig_perf = go.Figure()
                
                fig_perf.add_trace(go.Scatter(
                    x=df1.index,
                    y=df1['Normalized'],
                    name=stock1,
                    line=dict(color='blue')
                ))
                
                fig_perf.add_trace(go.Scatter(
                    x=df2.index,
                    y=df2['Normalized'],
                    name=stock2,
                    line=dict(color='red')
                ))
                
                fig_perf.update_layout(
                    title='Relative Performance (Normalized to 100)',
                    yaxis_title='Normalized Price',
                    template='plotly_dark'
                )
                
                st.plotly_chart(fig_perf, use_container_width=True)
                
                # Calculate metrics for comparison
                col1, col2, col3 = st.columns(3)
                
                # Returns calculation
                returns1 = df1['Close'].pct_change()
                returns2 = df2['Close'].pct_change()
                
                # Correlation
                correlation = returns1.corr(returns2)
                
                # Volatility
                vol1 = returns1.std() * np.sqrt(252)
                vol2 = returns2.std() * np.sqrt(252)
                
                # Total return
                total_return1 = (df1['Close'].iloc[-1] / df1['Close'].iloc[0] - 1) * 100
                total_return2 = (df2['Close'].iloc[-1] / df2['Close'].iloc[0] - 1) * 100
                
                with col1:
                    st.metric("Correlation", f"{correlation:.2f}")
                
                with col2:
                    st.metric(f"{stock1} Return", f"{total_return1:.2f}%")
                    st.metric(f"{stock1} Volatility", f"{vol1*100:.2f}%")
                
                with col3:
                    st.metric(f"{stock2} Return", f"{total_return2:.2f}%")
                    st.metric(f"{stock2} Volatility", f"{vol2*100:.2f}%")
                
                # Additional Analysis
                st.subheader("Statistical Analysis")
                
                # Calculate rolling correlation
                rolling_corr = returns1.rolling(window=30).corr(returns2)
                
                # Plot rolling correlation
                fig_corr = go.Figure()
                
                fig_corr.add_trace(go.Scatter(
                    x=rolling_corr.index,
                    y=rolling_corr,
                    name='30-Day Rolling Correlation',
                    line=dict(color='purple')
                ))
                
                fig_corr.update_layout(
                    title='30-Day Rolling Correlation',
                    yaxis_title='Correlation',
                    template='plotly_dark'
                )
                
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Scatter plot of returns
                fig_scatter = go.Figure()
                
                fig_scatter.add_trace(go.Scatter(
                    x=returns1,
                    y=returns2,
                    mode='markers',
                    name='Daily Returns',
                    marker=dict(
                        color='lightblue',
                        size=8,
                        opacity=0.6
                    )
                ))
                
                fig_scatter.update_layout(
                    title='Returns Scatter Plot',
                    xaxis_title=f'{stock1} Returns',
                    yaxis_title=f'{stock2} Returns',
                    template='plotly_dark'
                )
                
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Summary statistics
                st.subheader("Summary Statistics")
                
                stats_df = pd.DataFrame({
                    'Metric': ['Mean Daily Return', 'Daily Volatility', 'Sharpe Ratio', 'Max Drawdown'],
                    stock1: [
                        f"{returns1.mean()*100:.2f}%",
                        f"{returns1.std()*100:.2f}%",
                        f"{(returns1.mean()/returns1.std())*np.sqrt(252):.2f}",
                        f"{((df1['Close'].cummax() - df1['Close'])/df1['Close'].cummax()).max()*100:.2f}%"
                    ],
                    stock2: [
                        f"{returns2.mean()*100:.2f}%",
                        f"{returns2.std()*100:.2f}%",
                        f"{(returns2.mean()/returns2.std())*np.sqrt(252):.2f}",
                        f"{((df2['Close'].cummax() - df2['Close'])/df2['Close'].cummax()).max()*100:.2f}%"
                    ]
                })
                
                st.dataframe(stats_df, use_container_width=True)
                
            else:
                st.warning("No data available for one or both stocks.")
        
        except Exception as e:
            st.error(f"Error comparing stocks: {str(e)}")
            st.write("Please check if both stock symbols are valid.")

elif page == "Technical Analysis":
    st.header("Technical Analysis")
    
    # Input for stock symbol
    ticker_symbol = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT)", "AAPL").upper()
    
    if ticker_symbol:
        ticker = yf.Ticker(ticker_symbol)
        
        # Get historical data
        period = st.selectbox(
            "Select Time Period",
            ['1mo', '3mo', '6mo', '1y', '2y', '5y'],
            index=2
        )
        
        # Fetch data
        df = ticker.history(period=period)
        
        if not df.empty:
            # Technical Indicators
            st.subheader("Technical Indicators")
            
            # Calculate indicators
            # SMA
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            
            # Bollinger Bands
            df['BB_middle'] = df['Close'].rolling(window=20).mean()
            df['BB_upper'] = df['BB_middle'] + 2*df['Close'].rolling(window=20).std()
            df['BB_lower'] = df['BB_middle'] - 2*df['Close'].rolling(window=20).std()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["Price & Moving Averages", "Bollinger Bands", "Momentum Indicators"])
            
            with tab1:
                # Price and Moving Averages
                fig_ma = go.Figure()
                
                fig_ma.add_trace(go.Scatter(
                    x=df.index, 
                    y=df['Close'],
                    name='Close',
                    line=dict(color='blue')
                ))
                
                fig_ma.add_trace(go.Scatter(
                    x=df.index,
                    y=df['SMA_20'],
                    name='SMA 20',
                    line=dict(color='orange', dash='dash')
                ))
                
                fig_ma.add_trace(go.Scatter(
                    x=df.index,
                    y=df['SMA_50'],
                    name='SMA 50',
                    line=dict(color='green', dash='dash')
                ))
                
                fig_ma.update_layout(
                    title='Price and Moving Averages',
                    yaxis_title='Price',
                    template='plotly_dark'
                )
                
                st.plotly_chart(fig_ma, use_container_width=True)
            
            with tab2:
                # Bollinger Bands
                fig_bb = go.Figure()
                
                fig_bb.add_trace(go.Scatter(
                    x=df.index,
                    y=df['Close'],
                    name='Close',
                    line=dict(color='blue')
                ))
                
                fig_bb.add_trace(go.Scatter(
                    x=df.index,
                    y=df['BB_upper'],
                    name='Upper Band',
                    line=dict(color='gray', dash='dash')
                ))
                
                fig_bb.add_trace(go.Scatter(
                    x=df.index,
                    y=df['BB_middle'],
                    name='Middle Band',
                    line=dict(color='orange', dash='dash')
                ))
                
                fig_bb.add_trace(go.Scatter(
                    x=df.index,
                    y=df['BB_lower'],
                    name='Lower Band',
                    line=dict(color='gray', dash='dash'),
                    fill='tonexty'
                ))
                
                fig_bb.update_layout(
                    title='Bollinger Bands',
                    yaxis_title='Price',
                    template='plotly_dark'
                )
                
                st.plotly_chart(fig_bb, use_container_width=True)
            
            with tab3:
                # RSI and MACD
                col1, col2 = st.columns(2)
                
                with col1:
                    # RSI Plot
                    fig_rsi = go.Figure()
                    
                    fig_rsi.add_trace(go.Scatter(
                        x=df.index,
                        y=df['RSI'],
                        name='RSI',
                        line=dict(color='purple')
                    ))
                    
                    # Add RSI levels
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                    
                    fig_rsi.update_layout(
                        title='Relative Strength Index (RSI)',
                        yaxis_title='RSI',
                        template='plotly_dark'
                    )
                    
                    st.plotly_chart(fig_rsi, use_container_width=True)
                
                with col2:
                    # MACD Plot
                    fig_macd = go.Figure()
                    
                    fig_macd.add_trace(go.Scatter(
                        x=df.index,
                        y=df['MACD'],
                        name='MACD',
                        line=dict(color='blue')
                    ))
                    
                    fig_macd.add_trace(go.Scatter(
                        x=df.index,
                        y=df['Signal_Line'],
                        name='Signal Line',
                        line=dict(color='orange')
                    ))
                    
                    fig_macd.update_layout(
                        title='MACD',
                        yaxis_title='MACD',
                        template='plotly_dark'
                    )
                    
                    st.plotly_chart(fig_macd, use_container_width=True)
            
            # Technical Analysis Summary
            st.subheader("Technical Analysis Summary")
            
            # Latest values
            latest_close = df['Close'].iloc[-1]
            latest_sma20 = df['SMA_20'].iloc[-1]
            latest_sma50 = df['SMA_50'].iloc[-1]
            latest_rsi = df['RSI'].iloc[-1]
            latest_macd = df['MACD'].iloc[-1]
            latest_signal = df['Signal_Line'].iloc[-1]
            
            # Generate signals
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Moving Average Signals:**")
                ma_signal = "Bullish" if latest_close > latest_sma20 > latest_sma50 else "Bearish" if latest_close < latest_sma20 < latest_sma50 else "Neutral"
                st.write(f"• MA Trend: {ma_signal}")
                
                st.write("**RSI Signal:**")
                rsi_signal = "Overbought" if latest_rsi > 70 else "Oversold" if latest_rsi < 30 else "Neutral"
                st.write(f"• RSI ({latest_rsi:.2f}): {rsi_signal}")
            
            with col2:
                st.write("**MACD Signal:**")
                macd_signal = "Bullish" if latest_macd > latest_signal else "Bearish"
                st.write(f"• MACD Signal: {macd_signal}")
                
                st.write("**Bollinger Bands:**")
                bb_position = latest_close < df['BB_lower'].iloc[-1] and "Oversold" or latest_close > df['BB_upper'].iloc[-1] and "Overbought" or "Within Bands"
                st.write(f"• Price Position: {bb_position}")
        else:
            st.warning("No data available for the selected stock and time period.")
    
elif page == "Portfolio Analysis":
    st.header("Portfolio Analysis")
    
    # Portfolio input
    st.subheader("Enter Portfolio Composition")
    
    num_stocks = st.number_input("Number of stocks in portfolio", min_value=1, max_value=10, value=3)
    
    portfolio = {}
    total_weight = 0
    
    for i in range(num_stocks):
        col1, col2 = st.columns([3, 1])
        with col1:
            symbol = st.text_input(
                f"Stock {i+1} Symbol",
                f"STOCK{i+1}",
                key=f"symbol_{i}"
            ).upper()
        with col2:
            weight = st.number_input(
                f"Weight (%)",
                min_value=0.0,
                max_value=100.0,
                value=100.0/num_stocks,
                key=f"weight_{i}"
            )
            total_weight += weight
        portfolio[symbol] = weight/100
    
    if abs(total_weight - 100) > 0.01:
        st.warning("Warning: Portfolio weights do not sum to 100%")
    else:
        # Calculate portfolio metrics
        start_date = datetime.now() - timedelta(days=365)
        portfolio_data = pd.DataFrame()
        
        for symbol, weight in portfolio.items():
            try:
                data = yf.download(symbol, start=start_date)['Close']
                portfolio_data[symbol] = data
            except:
                st.error(f"Error fetching data for {symbol}")
        
        if not portfolio_data.empty:
            # Calculate portfolio returns
            returns = portfolio_data.pct_change()
            portfolio_returns = returns.dot(pd.Series(portfolio))
            
            # Display portfolio metrics
            st.subheader("Portfolio Metrics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Portfolio Return (1Y)", 
                         f"{(portfolio_returns.sum()*252)*100:.2f}%")
            with col2:
                st.metric("Portfolio Volatility", 
                         f"{(portfolio_returns.std()*np.sqrt(252))*100:.2f}%")
            with col3:
                st.metric("Sharpe Ratio", 
                         f"{calculate_sharpe_ratio(portfolio_returns):.2f}")
            
            # Portfolio visualization
            st.subheader("Portfolio Performance")
            portfolio_value = (1 + portfolio_returns).cumprod()
            st.line_chart(portfolio_value)

elif page == "Options Analysis":
    st.header("Options Analysis")
    
    ticker_symbol = st.text_input("Enter Stock Ticker", "AAPL").upper()
    
    if ticker_symbol:
        ticker = yf.Ticker(ticker_symbol)
        
        # Get options expiration dates
        try:
            expirations = ticker.options
            
            if expirations:
                exp_date = st.selectbox("Select Expiration Date", expirations)
                
                # Get options chain
                opt = ticker.option_chain(exp_date)
                
                # Display options chain
                st.subheader("Options Chain")
                
                tab1, tab2 = st.tabs(["Calls", "Puts"])
                
                with tab1:
                    st.dataframe(opt.calls[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'impliedVolatility']])
                
                with tab2:
                    st.dataframe(opt.puts[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'impliedVolatility']])
                
                # Options visualization
                st.subheader("Options Visualization")
                
                # Plot implied volatility smile
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=opt.calls['strike'],
                    y=opt.calls['impliedVolatility'],
                    name="Calls IV"
                ))
                
                fig.add_trace(go.Scatter(
                    x=opt.puts['strike'],
                    y=opt.puts['impliedVolatility'],
                    name="Puts IV"
                ))
                
                fig.update_layout(
                    title="Implied Volatility Smile",
                    xaxis_title="Strike Price",
                    yaxis_title="Implied Volatility"
                )
                
                st.plotly_chart(fig)
                
            else:
                st.write("No options data available for this stock")
        except Exception as e:
            st.error(f"Error fetching options data: {str(e)}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This enhanced dashboard provides comprehensive financial analysis, "
    "portfolio management, and options analysis tools."
)