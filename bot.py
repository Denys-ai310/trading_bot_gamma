import pandas as pd
import numpy as np
from datetime import datetime, timezone
import time
from pybit.unified_trading import HTTP
import telegram
import logging
import json
import os
from typing import Dict, List, Tuple
import asyncio
import math

# Set environment variables before importing TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # This will suppress INFO messages

import tensorflow as tf
from stable_baselines3 import PPO

# Configure TensorFlow logging
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
UTC = timezone.utc

class TradingBot:
    def __init__(self):
        # Bybit API credentials
        self.api_key = "oz6HRcOaFs1rb0TOEj"
        self.api_secret = "KxuD5pVBCJW9aMLCX3Vd5uY3Hm2MEZvMuUP6"
        
        # Telegram credentials
        self.telegram_token = "7382067227:AAGniEN3og9wt8B49RT3LbQ8E31up8tYYqs"
        self.telegram_chat_id = "-1002433151362"
        self.telegram_thread_id = 110  # Default thread ID
        
        # Trading parameters
        self.symbol = "BTCUSDT"
        self.leverage = 1
        self.take_profit_pct = 100  # 100% TP
        # self.take_profit_pct = 0.1
        self.stop_loss_pct = 4      # 4% SL
        # self.stop_loss_pct = 0.1
        self.timeframe = "D"        # Daily timeframe
        self.window_size = 32       # As per your trained model
        
        # Initialize connections
        self.bybit_client = HTTP(
            testnet=False,
            api_key=self.api_key,
            api_secret=self.api_secret
        )
        self.telegram_bot = telegram.Bot(token=self.telegram_token)
        
        # Load the trained model with custom objects
        custom_objects = {
            "learning_rate": lambda _: 0.0005,  # Default learning rate for PPO
            "lr_schedule": lambda _: 0.0005,
            "clip_range": lambda _: 0.2  # Default clip range for PPO
        }
        self.model = PPO.load("BTC_24h_sb3_ppo_32_0_26May24_147_58.zip", custom_objects=custom_objects)
        
        # Initialize trade tracking
        self.current_position = None
        self.trade_history = []

    async def send_telegram_message(self, message: str):
        """Send message to Telegram"""
        try:
            await self.telegram_bot.send_message(
                chat_id=self.telegram_chat_id,
                text=message,
                message_thread_id=self.telegram_thread_id
            )
            logging.info(f"Telegram message sent: {message}")
        except Exception as e:
            logging.error(f"Error sending Telegram message: {e}")

    def get_historical_data(self) -> pd.DataFrame:
        """Fetch historical data from Bybit"""
        max_retries = 5
        retry_delay = 3  # seconds
        
        for attempt in range(max_retries):
            try:
                # Add timeout to the API call
                klines = self.bybit_client.get_kline(
                    category="linear",
                    symbol=self.symbol,
                    interval=self.timeframe,
                    limit=self.window_size
                )
                
                df = pd.DataFrame(klines['result']['list'])
                df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']
                df = df.astype(float)
                return df
                
            except Exception as e:
                if "NameResolutionError" in str(e):
                    error_msg = f"DNS resolution failed (attempt {attempt + 1}/{max_retries}). Check your internet connection."
                elif "ConnectionError" in str(e):
                    error_msg = f"Connection error (attempt {attempt + 1}/{max_retries}). The API might be down."
                else:
                    error_msg = f"Error fetching data (attempt {attempt + 1}/{max_retries}): {str(e)}"
                
                logging.error(error_msg)
                
                if attempt < max_retries - 1:
                    logging.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    # Increase delay for next attempt
                    retry_delay *= 2
                else:
                    logging.error("Max retries reached while fetching historical data")
                    return None

    def prepare_model_input(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare data for model prediction"""
        try:
            # Make sure we have enough data
            if len(df) < self.window_size:
                raise ValueError(f"Not enough data points. Need {self.window_size}, got {len(df)}")
            
            # Convert strings to float if needed
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Calculate features for each day in the window
            features = []
            for i in range(self.window_size):
                # Basic price features
                features.extend([
                    df['open'].iloc[i],
                    df['high'].iloc[i],
                    df['low'].iloc[i],
                    df['close'].iloc[i],
                    df['volume'].iloc[i],
                    df['close'].iloc[i] / df['open'].iloc[i] - 1,  # Daily return
                    df['high'].iloc[i] / df['low'].iloc[i] - 1,    # Daily range
                ])
                
                # Moving averages
                if i >= 5:
                    sma5 = df['close'].iloc[i-5:i].mean()
                    features.append(df['close'].iloc[i] / sma5 - 1)
                else:
                    features.append(0)
                    
                if i >= 10:
                    sma10 = df['close'].iloc[i-10:i].mean()
                    features.append(df['close'].iloc[i] / sma10 - 1)
                else:
                    features.append(0)
                    
                # Volatility
                if i >= 10:
                    vol = df['close'].iloc[i-10:i].std() / df['close'].iloc[i]
                    features.append(vol)
                else:
                    features.append(0)
                
                # Volume features
                if i >= 5:
                    vol_sma5 = df['volume'].iloc[i-5:i].mean()
                    features.append(df['volume'].iloc[i] / vol_sma5 - 1)
                else:
                    features.append(0)
                    
                # Price momentum
                if i >= 3:
                    momentum = df['close'].iloc[i] / df['close'].iloc[i-3] - 1
                    features.append(momentum)
                else:
                    features.append(0)
                    
                # Additional feature: Typical price
                typical_price = (df['high'].iloc[i] + df['low'].iloc[i] + df['close'].iloc[i]) / 3
                features.append(typical_price / df['close'].iloc[i] - 1)  # Normalized typical price
            
            # Convert to numpy array
            features = np.array(features)
            
            # Ensure we have the correct shape (416,)
            if len(features) != 416:
                raise ValueError(f"Feature vector has incorrect shape. Expected 416, got {len(features)}")
            
            return features
            
        except Exception as e:
            logging.error(f"Error preparing model input: {e}")
            return np.array([])  # Return empty array on error

    def place_order(self, direction: str, quantity: float):
        """Place order on Bybit"""
        try:
            logging.info("Starting place_order method")
            # Calculate entry price and TP/SL levels
            ticker = self.bybit_client.get_tickers(
                category="linear",
                symbol=self.symbol
            )
            current_price = float(ticker['result']['list'][0]['lastPrice'])
            
            if direction == "long":
                take_profit = round(current_price * (1 + self.take_profit_pct/100), 1)
                stop_loss = round(current_price * (1 - self.stop_loss_pct/100), 1)
            else:  # short position
                # For short, take profit should be lower than current price
                take_profit = round(current_price * 0.5, 1)  # Set take profit at 50% of current price
                stop_loss = round(current_price * (1 + self.stop_loss_pct/100), 1)

            print("eeeeeeeeeeeeeeeee")
            print("take_profit:", take_profit)
            print("stop_loss:", stop_loss)
            if direction == "long":
                _side = "Buy"     
            else:
                _side = "Sell"

            quantity = math.floor(quantity * 1e3) / 1e3
            print("quantity:", quantity)
            
            # Place the order
            order = self.bybit_client.place_order(
                category="linear",
                symbol=self.symbol,
                side=_side,
                orderType="Market",
                qty=quantity,
                price=current_price,
                takeProfit=take_profit,
                stopLoss=stop_loss
            )
            logging.info(f"Order placed successfully: {order}") 
            
            # Calculate initial position value
            initial_position = quantity*current_price
            
            # Log and notify
            order_info = {
                "timestamp": datetime.now().isoformat(),
                "direction": direction,
                "entry_price": current_price,
                "quantity": quantity,
                "take_profit": take_profit,
                "stop_loss": stop_loss,
                "initial_position": initial_position
            }
            self.trade_history.append(order_info)
            self.current_position = order_info
            
            # Format message according to the new template
            message = (f"Trade Order Executed\n\n"
                      f"Model: PPO\n"
                      f"Direction: {'Long 📈' if direction == 'long' else 'Short 📉'}\n"
                      f"Pair: {self.symbol}\n"
                      f"Leverage: {self.leverage}x\n"
                      f"Initial Position: ${initial_position:.2f}\n\n"
                      f"Trade Breakdown:\n\n"
                      f"• Entry: ${current_price:.1f}\n"
                      f"• TP: ${take_profit:.1f}\n"
                      f"• SL: ${stop_loss:.1f}\n")
            logging.info(message)          
            

            # Start monitoring for TP/SL
            print("starting trade monitoring")
            logging.info("Starting trade monitoring")
            asyncio.create_task(self.monitor_trade(take_profit, stop_loss))
            return
            
        except Exception as e:
            logging.error(f"Error placing order: {e}")

    async def monitor_trade(self, take_profit: float, stop_loss: float):
        """Monitor the trade for take profit or stop loss conditions"""
        logging.info("Entering monitor_trade method")
        print("entering monitor_trade method")
        while self.current_position is not None:
            try:
                logging.info("Checking trade status")
                print("checking trade status")
                # Get current price
                ticker = self.bybit_client.get_tickers(
                    category="linear",
                    symbol=self.symbol
                )
                current_price = float(ticker['result']['list'][0]['lastPrice'])
                
                # Check for take profit or stop loss
                if self.current_position["direction"] == "long":
                    if current_price >= take_profit:
                        await self.notify_trade_closure(current_price, "TP Reached")
                        break
                    elif current_price <= stop_loss:
                        await self.notify_trade_closure(current_price, "SL Reached")
                        break
                else:  # short position
                    if current_price <= take_profit:
                        await self.notify_trade_closure(current_price, "TP Reached")
                        break
                    elif current_price >= stop_loss:
                        await self.notify_trade_closure(current_price, "SL Reached")
                        break
                logging.info("Waiting 5 minutes before next check")
                print("waiting 5 minutes before next check")
                # Wait before checking again
                await asyncio.sleep(300)  # Check every 300 seconds
                
            except Exception as e:
                logging.error(f"Error monitoring trade: {e}")
                await asyncio.sleep(300)  # Wait before retrying

    def generate_weekly_report(self):
        """Generate weekly performance report"""       
            
        # Calculate profit for each trade
        for trade in self.trade_history:
            if 'exit_price' in trade and 'entry_price' in trade:
                if trade['direction'] == 'long':
                    trade['profit'] = (trade['exit_price'] / trade['entry_price'] - 1) * 100
                else:  # short position
                    trade['profit'] = (trade['entry_price'] / trade['exit_price'] - 1) * 100
            else:
                trade['profit'] = 0  # If trade is still open
                
        df = pd.DataFrame(self.trade_history)
        
        # Only consider completed trades (those with profit calculated)
        completed_trades = df[df['profit'] != 0]
        
        weekly_stats = {
            "total_trades": len(completed_trades),
            "winning_trades": len(completed_trades[completed_trades['profit'] > 0]),
            "losing_trades": len(completed_trades[completed_trades['profit'] < 0]),
            "total_profit": completed_trades['profit'].sum(),
            "win_rate": len(completed_trades[completed_trades['profit'] > 0]) / len(completed_trades) * 100 if len(completed_trades) > 0 else 0
        }
        
        report = (f"Weekly Trading Report\n\n"
                 f"Total Trades: {weekly_stats['total_trades']}\n"
                 f"Win Rate: {weekly_stats['win_rate']:.2f}%\n"
                 f"Total Profit: {weekly_stats['total_profit']:.2f}%\n"
                 f"Winning Trades: {weekly_stats['winning_trades']}\n"
                 f"Losing Trades: {weekly_stats['losing_trades']}")
                 
        return report

    async def notify_trade_closure(self, exit_price: float, reason: str):
        """Send notification when a trade is closed"""
        if not self.current_position:
            return
            
        entry_price = self.current_position["entry_price"]
        initial_position = self.current_position["initial_position"]
        direction = self.current_position["direction"]
        
        # Calculate final position and PNL
        final_position = initial_position * (exit_price / entry_price)
        pnl = final_position - initial_position
        pnl_percentage = (pnl / initial_position) * 100
        
        # Assume 0.1% fee per trade (entry + exit)
        fees = initial_position * 0.002
        pnl_no_fees = pnl + fees
        pnl_no_fees_percentage = (pnl_no_fees / initial_position) * 100
        
        # Update the last trade in trade_history with exit information
        if self.trade_history:
            last_trade = self.trade_history[-1]
            last_trade.update({
                "exit_price": exit_price,
                "reason": reason,
                "pnl": pnl,
                "pnl_percentage": pnl_percentage
            })
        
        message = (f"Trade Order Executed\n\n"
                  f"Model: PPO\n"
                  f"Direction: {'Long 📈' if direction == 'long' else 'Short 📉'}\n"
                  f"Pair: {self.symbol}\n"
                  f"Leverage: {self.leverage}x\n"
                  f"Initial Position: ${initial_position:.2f}\n"
                  f"Order Closed: ${final_position:.2f} ({pnl_percentage:.2f}%)\n"
                  f"Reason: {reason}\n\n"
                  f"Trade Breakdown:\n\n"
                  f"• Entry: ${entry_price:.1f}\n"
                  f"• Exit: ${exit_price:.1f}\n"
                  f"• PNL: {'+' if pnl >= 0 else ''}${pnl:.2f} ({'+' if pnl_percentage >= 0 else ''}{pnl_percentage:.2f}%)\n"
                  f"• PNL (No Fees): {'+' if pnl_no_fees >= 0 else ''}${pnl_no_fees:.2f} ({'+' if pnl_no_fees_percentage >= 0 else ''}{pnl_no_fees_percentage:.2f}%)")
        
        await self.send_telegram_message(message)
        
        # Clear current position after closure
        self.current_position = None

    def close_all_positions(self):
        """Close all open positions"""
        try:
            # Get all open positions
            positions = self.bybit_client.get_positions(
                category="linear",
                symbol=self.symbol
            )
            
            if positions and 'result' in positions and 'list' in positions['result']:
                for position in positions['result']['list']:
                    if float(position['size']) > 0:  # If there's an open position
                        side = "Sell" if position['side'] == "Buy" else "Buy"  # Opposite side to close
                        quantity = float(position['size'])
                        
                        # Place closing order
                        order = self.bybit_client.place_order(
                            category="linear",
                            symbol=self.symbol,
                            side=side,
                            orderType="Market",
                            qty=quantity,
                            reduceOnly=True  # This ensures it only closes positions
                        )
                        
                        message = f"Closed position:\nSide: {position['side']}\nSize: {quantity}\nEntry Price: {position['avgPrice']}"
                        logging.info(f"Closed position: {message}")
            
            message = "All positions have been closed"
            logging.info(message)
            return
            
        except Exception as e:
            error_msg = f"Error closing positions: {str(e)}"
            logging.error(error_msg)
            return
        

    async def run(self):
        """Main trading loop"""
        try:
            # Send welcome message when bot starts
            welcome_message = (
                "Good day everyone, we have now paused the TA trade\n"
                "notification and will be going live with the Gamma model which\n"
                "will be opening a trade on either a long or short position every\n"
                "24hr at 00:00UTC"
            )
            await self.send_telegram_message(welcome_message)

            while True:
                current_time = datetime.now(UTC)
                if current_time.hour == 0 and current_time.minute == 0:
                
                    if self.trade_history:
                        if current_time.weekday() == 6:
                            
                        # if (current_time.minute % 3 == 0):
                            report = self.generate_weekly_report()
                            await self.send_telegram_message(report)                                                
                            continue 
                        
                    # Get historical data
                    df = self.get_historical_data()
                    if df is not None:
                        # Prepare model input
                        model_input = self.prepare_model_input(df)
                        
                        # Get prediction
                        prediction = self.model.predict(model_input)
                        direction = "long" if prediction[0] > 0 else "short"
                        
                        # Get account balance and calculate position size
                        account_info = self.bybit_client.get_wallet_balance(
                            accountType="UNIFIED"
                        )
                        print("account_info:", account_info)
                        logging.info(f"account_info: {account_info}")

                        ticker = self.bybit_client.get_tickers(
                            category="linear",
                            symbol=self.symbol
                        )
                        current_price = float(ticker['result']['list'][0]['lastPrice'])
                        
                        # Initialize balances
                        btc_balance = 0
                        usdt_balance = 0
                        MIN_USDT_QTY = 1  # Minimum BTC order quantity
                        MAX_USDT_QTY = 8000000     # Maximum BTC order quantity
                        MIN_BTC_QTY = round(MIN_USDT_QTY/current_price * 1e6) / 1e6
                        MAX_BTC_QTY = math.floor(MAX_USDT_QTY/current_price * 1e6) / 1e6
                        
                        if account_info and 'result' in account_info and 'list' in account_info['result']:
                            if direction == "long":
                                wallet_info = next((coin for coin in account_info['result']['list'][0]['coin'] if coin['coin'] == 'USDT'), None)
                                if wallet_info:
                                    usdt_balance = float(wallet_info['walletBalance'])
                                else:
                                    usdt_balance = 0

                                if usdt_balance < MIN_USDT_QTY:
                                    message = f"Predicted Direction: {'Long 📈' if direction == 'long' else 'Short 📉'}\nInsufficient USDT balance.\nHave: {usdt_balance}\nNeed minimum: {MIN_USDT_QTY}"
                                    logging.error(message)
                                    await self.send_telegram_message(message)
                                    await asyncio.sleep(86395)  
                                    continue
                                    
                                balance = usdt_balance/current_price
                                balance = min(balance, MAX_BTC_QTY)
                                balance = math.floor(balance * 1e5) / 1e5
                            else:
                                wallet_info = next((coin for coin in account_info['result']['list'][0]['coin'] if coin['coin'] == 'BTC'), None)
                                if wallet_info:
                                    btc_balance = float(wallet_info['walletBalance'])
                                else:
                                    btc_balance = 0
                                
                                if btc_balance < MIN_BTC_QTY:
                                    message = f"Predicted Direction: {'Long 📈' if direction == 'long' else 'Short 📉'}\nInsufficient BTC balance.\nHave: {btc_balance}\nNeed minimum: {MIN_BTC_QTY}"
                                    logging.error(message)
                                    await self.send_telegram_message(message)
                                    await asyncio.sleep(86395)  
                                    continue
                                    
                                balance = min(btc_balance, MAX_BTC_QTY)
                                balance = math.floor(balance * 1e5) / 1e5
                            
                            # Place the trade
                            self.place_order(direction, balance)
                            
                            # Wait for 1 minute
                            await asyncio.sleep(86395) 
                            
                            # Close all positions
                            self.close_all_positions()
                            
                            # Small delay before next iteration
                            await asyncio.sleep(1)
                
        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            return  # Exit on error

if __name__ == "__main__":
    bot = TradingBot()
    bot.close_all_positions()
    asyncio.run(bot.run())