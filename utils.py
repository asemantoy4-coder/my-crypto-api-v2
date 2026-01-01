def combined_technical_analysis(data: List, symbol: str = "BTCUSDT", timeframe: str = "5m") -> Dict[str, Any]:
    """
    تحلیل تکنیکال ترکیبی با تمام اندیکاتورها - نسخه کامل
    
    پارامترها:
    -----------
    data : List
        لیست کندل‌ها در فرمت استاندارد
    symbol : str
        نماد ارز (پیش‌فرض: BTCUSDT)
    timeframe : str
        تایم‌فریم تحلیل (پیش‌فرض: 5m)
    
    بازگشت:
    -------
    Dict[str, Any]
        دیکشنری شامل تحلیل کامل تکنیکال
    """
    try:
        if not data or len(data) < 50:
            return {
                "status": "error",
                "message": "داده ناکافی برای تحلیل (نیاز به حداقل ۵۰ کندل)",
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": pd.Timestamp.now().isoformat()
            }
        
        logger.info(f"Starting combined analysis for {symbol} ({timeframe}) - {len(data)} candles")
        
        # ==========================================================================
        # 1. محاسبه اندیکاتورهای اصلی
        # ==========================================================================
        
        # قیمت فعلی
        current_price = float(data[-1][4]) if len(data[-1]) > 4 else 0
        
        # محاسبه RSI
        rsi = calculate_simple_rsi(data, 14)
        rsi_status = "خرید هیجانی" if rsi > 70 else "فروش هیجانی" if rsi < 30 else "معمولی"
        
        # محاسبه SMA
        sma_20 = calculate_simple_sma(data, 20)
        sma_50 = calculate_simple_sma(data, 50)
        sma_position = "قیمت بالای SMA" if current_price > sma_20 else "قیمت زیر SMA"
        
        # محاسبه MACD
        macd_result = calculate_macd(data)
        macd_trend = macd_result.get('trend', 'neutral')
        macd_divergence = macd_result.get('divergence')
        
        # محاسبه ایچیموکو
        ichimoku = calculate_ichimoku_components(data)
        ichimoku_signal = analyze_ichimoku_scalp_signal(ichimoku) if ichimoku else {
            "signal": "HOLD", 
            "confidence": 0.5,
            "reason": "ایچیموکو محاسبه نشد",
            "details": {}
        }
        
        # محاسبه سطوح حمایت و مقاومت
        sr_levels = get_support_resistance_levels(data)
        
        # محاسبه نوسان
        volatility = calculate_volatility(data, 20)
        volatility_status = "بالا" if volatility > 2.0 else "متوسط" if volatility > 1.0 else "پایین"
        
        # ==========================================================================
        # 2. سیستم امتیازدهی برای سیگنال‌ها
        # ==========================================================================
        
        signals = {
            'buy': 0.0,
            'sell': 0.0,
            'hold': 0.0
        }
        
        # امتیازدهی RSI
        if rsi < 30:
            signals['buy'] += 2.5
            logger.debug(f"RSI oversold: +2.5 for BUY (RSI: {rsi:.1f})")
        elif rsi > 70:
            signals['sell'] += 2.5
            logger.debug(f"RSI overbought: +2.5 for SELL (RSI: {rsi:.1f})")
        else:
            signals['hold'] += 1.0
        
        # امتیازدهی ایچیموکو
        ich_signal = ichimoku_signal.get('signal', 'HOLD')
        ich_confidence = ichimoku_signal.get('confidence', 0.5)
        
        if ich_signal == 'BUY':
            signals['buy'] += ich_confidence * 3.0
            logger.debug(f"Ichimoku BUY: +{ich_confidence*3:.2f} for BUY")
        elif ich_signal == 'SELL':
            signals['sell'] += ich_confidence * 3.0
            logger.debug(f"Ichimoku SELL: +{ich_confidence*3:.2f} for SELL")
        
        # امتیازدهی میانگین‌های متحرک
        if current_price > sma_20 > sma_50:
            signals['buy'] += 2.0
            logger.debug(f"Golden Cross: +2.0 for BUY")
        elif current_price < sma_20 < sma_50:
            signals['sell'] += 2.0
            logger.debug(f"Death Cross: +2.0 for SELL")
        
        # امتیازدهی MACD
        if macd_trend == 'bullish':
            signals['buy'] += 1.5
            logger.debug(f"MACD bullish: +1.5 for BUY")
        elif macd_trend == 'bearish':
            signals['sell'] += 1.5
            logger.debug(f"MACD bearish: +1.5 for SELL")
        
        # امتیازدهی واگرایی MACD
        if macd_divergence == 'bullish_divergence':
            signals['buy'] += 1.0
            logger.debug(f"MACD bullish divergence: +1.0 for BUY")
        elif macd_divergence == 'bearish_divergence':
            signals['sell'] += 1.0
            logger.debug(f"MACD bearish divergence: +1.0 for SELL")
        
        # امتیازدهی موقعیت قیمت نسبت به سطوح
        support = sr_levels.get('support', 0)
        resistance = sr_levels.get('resistance', 0)
        
        if support > 0 and current_price < support * 1.02:  # نزدیک به حمایت (2%)
            signals['buy'] += 1.0
            logger.debug(f"Near support: +1.0 for BUY")
        
        if resistance > 0 and current_price > resistance * 0.98:  # نزدیک به مقاومت (2%)
            signals['sell'] += 1.0
            logger.debug(f"Near resistance: +1.0 for SELL")
        
        # ==========================================================================
        # 3. تصمیم‌گیری نهایی
        # ==========================================================================
        
        # تشخیص سیگنال برنده
        final_signal = max(signals, key=signals.get)
        total_score = sum(signals.values())
        
        if total_score > 0:
            confidence = signals[final_signal] / total_score
        else:
            confidence = 0.5
        
        # تنظیم آستانه اعتماد
        min_confidence = 0.6
        if confidence < min_confidence:
            final_signal = 'hold'
            confidence = 0.5
        
        # محاسبه ورود هوشمند
        smart_entry = calculate_smart_entry(data, final_signal.upper())
        if smart_entry <= 0:
            smart_entry = current_price
        
        # ==========================================================================
        # 4. محاسبه تارگت و استاپ لاس
        # ==========================================================================
        
        stop_loss = 0
        targets = []
        
        if final_signal == 'buy':
            # برای خرید: استاپ زیر حمایت یا 1.5% پایین‌تر از ورود
            stop_loss_candidate1 = sr_levels.get('support', smart_entry * 0.985)
            stop_loss_candidate2 = smart_entry * 0.985  # 1.5% پایین‌تر
            stop_loss = round(min(stop_loss_candidate1, stop_loss_candidate2), 4)
            
            # تارگت‌ها: 1.5% و 3% بالاتر
            targets = [
                round(smart_entry * 1.015, 4),  # Target 1: +1.5%
                round(smart_entry * 1.03, 4)    # Target 2: +3.0%
            ]
            
            logger.debug(f"BUY signal - Entry: {smart_entry:.4f}, Stop: {stop_loss:.4f}, "
                        f"Targets: {targets}")
            
        elif final_signal == 'sell':
            # برای فروش: استاپ بالای مقاومت یا 1.5% بالاتر از ورود
            stop_loss_candidate1 = sr_levels.get('resistance', smart_entry * 1.015)
            stop_loss_candidate2 = smart_entry * 1.015  # 1.5% بالاتر
            stop_loss = round(max(stop_loss_candidate1, stop_loss_candidate2), 4)
            
            # تارگت‌ها: 1.5% و 3% پایین‌تر
            targets = [
                round(smart_entry * 0.985, 4),  # Target 1: -1.5%
                round(smart_entry * 0.97, 4)    # Target 2: -3.0%
            ]
            
            logger.debug(f"SELL signal - Entry: {smart_entry:.4f}, Stop: {stop_loss:.4f}, "
                        f"Targets: {targets}")
        
        else:  # HOLD
            stop_loss = 0
            targets = []
            logger.debug("HOLD signal - No position recommended")
        
        # ==========================================================================
        # 5. جمع‌بندی و بازگشت نتیجه
        # ==========================================================================
        
        # آماده‌سازی دلایل سیگنال
        reasons = []
        
        if rsi < 30:
            reasons.append(f"RSI در ناحیه فروش هیجانی ({rsi:.1f})")
        elif rsi > 70:
            reasons.append(f"RSI در ناحیه خرید هیجانی ({rsi:.1f})")
        
        if ichimoku_signal.get('reason'):
            reasons.append(ichimoku_signal['reason'])
        
        if current_price > sma_20 > sma_50:
            reasons.append("الگوی گلدن کراس (قیمت > SMA20 > SMA50)")
        elif current_price < sma_20 < sma_50:
            reasons.append("الگوی دث کراس (قیمت < SMA20 < SMA50)")
        
        if macd_divergence:
            reasons.append(f"واگرایی {macd_divergence.replace('_', ' ')} در MACD")
        
        # اطلاعات جزئی اندیکاتورها
        indicator_details = {
            "rsi": {
                "value": round(rsi, 2),
                "status": rsi_status,
                "overbought": rsi > 70,
                "oversold": rsi < 30
            },
            "sma": {
                "sma_20": round(sma_20, 4),
                "sma_50": round(sma_50, 4),
                "position": sma_position,
                "distance_from_sma_20": round(((current_price - sma_20) / sma_20 * 100), 2) if sma_20 > 0 else 0
            },
            "macd": {
                "trend": macd_trend,
                "divergence": macd_divergence,
                "line": round(macd_result.get('macd_line', 0), 4),
                "signal": round(macd_result.get('signal_line', 0), 4),
                "histogram": round(macd_result.get('histogram', 0), 4)
            },
            "ichimoku": ichimoku_signal.get('details', {}),
            "volatility": {
                "value": round(volatility, 2),
                "status": volatility_status
            }
        }
        
        # نتیجه نهایی
        result = {
            "status": "success",
            "symbol": symbol,
            "timeframe": timeframe,
            "signal": final_signal.upper(),
            "confidence": round(confidence, 2),
            "current_price": round(current_price, 4),
            "entry_price": round(smart_entry, 4),
            "stop_loss": stop_loss,
            "targets": targets,
            "risk_reward_ratio": round((targets[0] - smart_entry) / (smart_entry - stop_loss), 2) if targets and stop_loss > 0 and smart_entry > stop_loss else 0,
            "analysis_summary": {
                "signal_score": {
                    "buy": round(signals['buy'], 2),
                    "sell": round(signals['sell'], 2),
                    "hold": round(signals['hold'], 2)
                },
                "reasons": reasons,
                "market_condition": "روندی" if volatility > 1.5 else "رنج"
            },
            "support_resistance": {
                "support": round(sr_levels.get('support', 0), 4),
                "resistance": round(sr_levels.get('resistance', 0), 4),
                "strong_support": round(sr_levels.get('strong_support', 0), 4),
                "strong_resistance": round(sr_levels.get('strong_resistance', 0), 4)
            },
            "indicators": indicator_details,
            "timestamp": pd.Timestamp.now().isoformat(),
            "data_points": len(data)
        }
        
        # لاگ نتیجه
        if final_signal != 'hold':
            logger.info(f"Analysis COMPLETE for {symbol}: {final_signal.upper()} "
                       f"(Confidence: {confidence:.2f}, Entry: {smart_entry:.4f})")
        else:
            logger.info(f"Analysis COMPLETE for {symbol}: HOLD (Confidence: {confidence:.2f})")
        
        return result
        
    except Exception as e:
        logger.error(f"Critical error in combined analysis for {symbol}: {e}", exc_info=True)
        return {
            "status": "error",
            "message": f"خطا در تحلیل: {str(e)}",
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": pd.Timestamp.now().isoformat()
        }


def calculate_volatility(data: List, period: int = 20) -> float:
    """
    محاسبه نوسان قیمت
    
    پارامترها:
    -----------
    data : List
        لیست کندل‌ها
    period : int
        دوره محاسبه
    
    بازگشت:
    -------
    float
        نوسان به درصد
    """
    try:
        if not data or len(data) < period:
            return 0.0
        
        closes = []
        for candle in data[-period:]:
            if len(candle) > 4:
                try:
                    closes.append(float(candle[4]))
                except:
                    continue
        
        if len(closes) < 2:
            return 0.0
        
        # محاسبه بازده روزانه
        returns = []
        for i in range(1, len(closes)):
            if closes[i-1] > 0:
                daily_return = (closes[i] - closes[i-1]) / closes[i-1]
                returns.append(abs(daily_return))
        
        if not returns:
            return 0.0
        
        # نوسان به صورت انحراف معیار بازده‌ها
        volatility = np.std(returns) * 100
        
        return float(round(volatility, 2))
        
    except Exception as e:
        logger.error(f"Error calculating volatility: {e}")
        return 0.0


# تابع کمکی برای تست
def test_combined_analysis():
    """
    تست عملکرد تحلیل ترکیبی
    """
    print("Testing Combined Technical Analysis...")
    print("=" * 60)
    
    # تولید داده نمونه
    sample_data = []
    base_price = 50000.0
    
    for i in range(100):
        timestamp = 1700000000000 + i * 300000
        open_price = base_price * (1 + np.random.uniform(-0.01, 0.01))
        close_price = base_price * (1 + np.random.uniform(-0.02, 0.02))
        high_price = max(open_price, close_price) * (1 + np.random.uniform(0, 0.015))
        low_price = min(open_price, close_price) * (1 - np.random.uniform(0, 0.015))
        volume = np.random.uniform(10000, 50000)
        
        candle = [
            int(timestamp),
            float(open_price),
            float(high_price),
            float(low_price),
            float(close_price),
            float(volume),
            int(timestamp + 300000),
            str(float(volume) * float(close_price)),
            str(np.random.randint(500, 2000)),
            str(float(volume) * 0.6),
            str(float(volume) * float(close_price) * 0.6),
            "0"
        ]
        sample_data.append(candle)
        base_price = close_price
    
    # اجرای تحلیل
    result = combined_technical_analysis(sample_data, "BTCUSDT", "5m")
    
    # نمایش نتایج
    if result["status"] == "success":
        print(f"✅ Analysis Successful!")
        print(f"Symbol: {result['symbol']}")
        print(f"Timeframe: {result['timeframe']}")
        print(f"Signal: {result['signal']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Current Price: ${result['current_price']:,.2f}")
        print(f"Entry Price: ${result['entry_price']:,.2f}")
        print(f"Stop Loss: ${result['stop_loss']:,.2f}")
        print(f"Targets: {[f'${t:,.2f}' for t in result['targets']]}")
        print(f"Risk/Reward Ratio: {result['risk_reward_ratio']}")
        
        print(f"\n📊 Indicators:")
        print(f"  RSI: {result['indicators']['rsi']['value']} ({result['indicators']['rsi']['status']})")
        print(f"  SMA20: ${result['indicators']['sma']['sma_20']:,.2f}")
        print(f"  SMA50: ${result['indicators']['sma']['sma_50']:,.2f}")
        print(f"  MACD Trend: {result['indicators']['macd']['trend']}")
        
        print(f"\n📈 Support/Resistance:")
        print(f"  Support: ${result['support_resistance']['support']:,.2f}")
        print(f"  Resistance: ${result['support_resistance']['resistance']:,.2f}")
        
        print(f"\n📝 Reasons:")
        for reason in result['analysis_summary']['reasons']:
            print(f"  • {reason}")
            
    else:
        print(f"❌ Analysis Failed: {result['message']}")
    
    print("\n" + "=" * 60)
    print("Test completed!")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    test_combined_analysis()