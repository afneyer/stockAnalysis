
class MyData:
    # data types
    quandle = 'quandle'
    multpl = 'multpl'
    compute = 'compute'
    fred = 'FRED'
    yahoo = 'yahoo'

    # define all data series ids
    sp500_pe_ratio_month = 'SP500_PE_Ratio_Month'
    sp500_div_yield_month = 'SP500_Div_Yield_Month'
    sp500_real_price_month = 'SP500_Real_Price_Month'
    sp500_pe_ratio_month_quandle = 'SP500_PE_Ratio_Month_Quandle'
    sp500_div_yield_month_quandle = 'SP500_Div_Yield_Month_Quandle'
    sp500_real_price_month_quandle = 'SP500_Real_Price_Month_Quandle'
    sp500_div_reinvest_month = 'SP500_Div_Reinvest_Month'
    sp500_earnings_growth = 'SP500_Growth_Based_On_Earnings'
    sp500_earnings_yield = "SP500_Earnings_Annual_Yield_Monthly"
    us_gdp_nominal = "Nominal_US_GDP_Quarterly"
    ten_year_minus_two_year = "10-Year Treasury Minus 2-Year Treasury Constant Maturity"
    int_one_month_cm = "1-Month Treasury Constant Maturity"
    int_three_month_cm = "3-Month Treasury Constant Maturity"
    int_six_month_cm = "6-Month Treasury Constant Maturity"
    int_one_year_cm = "1-Year Treasury Constant Maturity"
    int_two_year_cm = "2-Year Treasury Constant Maturity"
    int_five_year_cm = "5-Year Treasury Constant Maturity"
    int_ten_year_cm = "10-Year Treasury Constant Maturity"
    int_thirty_year_cm = "30-Year Treasury Constant Maturity"
    sp500_div_reinvest_day = "SP500 Dividends Reinvested Daily"
    mult_eco_consumer_price_index_cpi = 'mult_eco_consumer_price_index_cpi'
    mult_eco_us_average_income = 'mult_eco_us_average_income'
    mult_eco_us_federal_debt_percent = 'mult_eco_us_federal_debt_percent'
    mult_eco_us_gdp = 'mult_eco_us_gdp'
    mult_eco_us_gdp_growth_rate = 'mult_eco_us_gdp_growth_rate'
    mult_eco_us_home_prices = 'mult_eco_us_home_prices'
    mult_eco_us_inflation_rate = 'mult_eco_us_inflation_rate'
    mult_eco_us_median_income = 'mult_eco_us_median_income'
    mult_eco_us_median_income_growth = 'mult_eco_us_median_income_growth'
    mult_eco_us_median_real_income = 'mult_eco_us_median_real_income'
    mult_eco_us_population = 'mult_eco_us_population'
    mult_eco_us_population_growth_rate = 'mult_eco_us_population_growth_rate'
    mult_eco_us_real_gdp = 'mult_eco_us_real_gdp'
    mult_eco_us_real_gdp_growth_rate = 'mult_eco_us_real_gdp_growth_rate'
    mult_eco_us_real_gdp_per_capita = 'mult_eco_us_real_gdp_per_capita'
    mult_real_rate_10_year = 'mult_real_rate_10_year'
    mult_real_rate_20_year = 'mult_real_rate_20_year'
    mult_real_rate_30_year = 'mult_real_rate_30_year'
    mult_real_rate_5_year = 'mult_real_rate_5_year'
    mult_sp500_book_value_per_share = "mult_sp500_book_value_per_share"
    mult_sp500_dividend = "mult_sp500_dividend"
    mult_sp500_dividend_growth = "mult_sp500_dividend_growth"
    mult_sp500_dividend_yield = "mult_sp500_dividend_yield"
    mult_sp500_earnings = "mult_sp500_earnings"
    mult_sp500_earnings_growth = "mult_sp500_earnings_growth"
    mult_sp500_earnings_yield = "mult_sp500_earnings_yield"
    mult_sp500_historical_prices = "mult_sp500_historical_prices"
    mult_sp500_inflation_adjusted_prices = "mult_sp500_inflation_adjusted_prices"
    mult_sp500_pe_ratio = "mult_sp500_pe_ratio"
    mult_sp500_price_to_book_value = "mult_sp500_price_to_book_value"
    mult_sp500_price_to_sales_ratio = "mult_sp500_price_to_sales_ratio"
    mult_sp500_real_earnings_growth = "mult_sp500_real_earnings_growth"
    mult_sp500_real_sales_per_share = "mult_sp500_real_sales_per_share"
    mult_sp500_real_sales_per_share_growth = "mult_sp500_real_sales_per_share_growth"
    mult_sp500_sales_per_share = "mult_sp500_sales_per_share"
    mult_sp500_sales_per_share_growth = "mult_sp500_sales_per_share_growth"
    mult_sp500_shiller_pe_10_ratio = "mult_sp500_shiller_pe_10_ratio"
    mult_trate_1_month = "mult_trate_1month"
    mult_trate_1_year = 'mult_trate_1_year'
    mult_trate_10_year = 'mult_trate_10_year'
    mult_trate_2_year = 'mult_trate_2_year'
    mult_trate_20_year = 'mult_trate_20_year'
    mult_trate_3_year = 'mult_trate_3_year'
    mult_trate_30_year = 'mult_trate_30_year'
    mult_trate_5_year = 'mult_trate_5_year'
    mult_trate_6_month = 'mult_trate_6_month'

    urls = [
        [sp500_pe_ratio_month, quandle, 'MULTPL/SP500_PE_RATIO_MONTH'],
        [sp500_div_yield_month, quandle, 'MULTPL/SP500_DIV_YIELD_MONTH'],
        [sp500_real_price_month, quandle, 'MULTPL/SP500_REAL_PRICE_MONTH'],
        [us_gdp_nominal, fred, 'GDP'],
        [ten_year_minus_two_year, fred, 'T10Y2Y'],
        [int_one_month_cm, fred, 'DGS1MO'],
        [int_three_month_cm, fred, 'DGS3MO'],
        [int_six_month_cm, fred, 'DGS6MO'],
        [int_one_year_cm, fred, 'DGS1'],
        [int_two_year_cm, fred, 'DGS2'],
        [int_five_year_cm, fred, 'DGS5'],
        [int_ten_year_cm, fred, 'DGS10'],
        [int_thirty_year_cm, fred, 'DGS30'],
        [sp500_div_reinvest_day, yahoo, 'spy'],
        [mult_eco_consumer_price_index_cpi, multpl, 'https://www.multpl.com/cpi/table/by-month'],
        [mult_eco_us_average_income, multpl, 'https://www.multpl.com/us-average-income/table/by-month'],
        [mult_eco_us_federal_debt_percent, multpl, 'https://www.multpl.com/u-s-federal-debt-percent/table/by-month'],
        [mult_eco_us_gdp, multpl, 'https://www.multpl.com/us-gdp/table/by-month'],
        [mult_eco_us_gdp_growth_rate, multpl, 'https://www.multpl.com/us-real-gdp-growth-rate/table/by-month'],
        [mult_eco_us_home_prices, multpl, 'https://www.multpl.com/case-shiller-home-price-index-inflation-adjusted/table/by-month'],
        [mult_eco_us_inflation_rate, multpl, 'https://www.multpl.com/inflation/table/by-month'],
        [mult_eco_us_median_income, multpl, 'https://www.multpl.com/us-median-income/table/by-year'],
        [mult_eco_us_median_income_growth, multpl, 'https://www.multpl.com/us-median-income-growth/table/by-year'],
        [mult_eco_us_median_real_income, multpl, 'https://www.multpl.com/us-median-real-income/table/by-year'],
        [mult_eco_us_population, multpl, 'https://www.multpl.com/united-states-population/table/by-month'],
        [mult_eco_us_population_growth_rate, multpl, 'https://www.multpl.com/us-population-growth-rate/table/by-year'],
        [mult_eco_us_real_gdp, multpl, 'https://www.multpl.com/us-gdp-inflation-adjusted/table/by-quarter'],
        [mult_eco_us_real_gdp_growth_rate, multpl, 'https://www.multpl.com/us-real-gdp-growth-rate/table/by-quarter'],
        [mult_eco_us_real_gdp_per_capita, multpl, 'https://www.multpl.com/us-real-gdp-per-capita/table/by-quarter'],
        [mult_real_rate_10_year, multpl, 'https://www.multpl.com/10-year-treasury-rate/table/by-month'],
        [mult_real_rate_20_year, multpl, 'https://www.multpl.com/20-year-treasury-rate/table/by-month'],
        [mult_real_rate_30_year, multpl, 'https://www.multpl.com/30-year-treasury-rate/table/by-month'],
        [mult_real_rate_5_year, multpl, 'https://www.multpl.com/5-year-treasury-rate/table/by-month'],
        [mult_sp500_book_value_per_share, multpl, 'https://www.multpl.com/s-p-500-book-value/table/by-quarter'],
        [mult_sp500_dividend, multpl, 'https://www.multpl.com/s-p-500-dividend/table/by-month'],
        [mult_sp500_dividend_growth, multpl, 'https://www.multpl.com/s-p-500-dividend-growth/table/by-month'],
        [mult_sp500_dividend_yield, multpl, 'https://www.multpl.com/s-p-500-dividend-yield/table/by-month'],
        [mult_sp500_earnings, multpl, 'https://www.multpl.com/s-p-500-earnings/table/by-month'],
        [mult_sp500_earnings_yield, multpl, 'https://www.multpl.com/s-p-500-earnings-yield/table/by_quarter'],
        [mult_sp500_earnings_growth, multpl, 'https://www.multpl.com/s-p-500-earnings-growth/table/by-quarter'],
        [mult_sp500_real_earnings_growth, multpl, 'https://www.multpl.com/s-p-500-real-earnings-growth/table/by-quarter'],
        [mult_sp500_pe_ratio, multpl, 'https://www.multpl.com/s-p-500-pe-ratio/table/by-month'],
        [mult_sp500_historical_prices, multpl, 'https://www.multpl.com/s-p-500-historical-prices/table/by-month'],
        [mult_sp500_inflation_adjusted_prices, multpl, 'https://www.multpl.com/inflation-adjusted-s-p-500/table/by-year'],
        [mult_sp500_price_to_book_value, multpl, 'https://www.multpl.com/s-p-500-price-to-book/table/by-year'],
        [mult_sp500_price_to_sales_ratio, multpl, 'https://www.multpl.com/s-p-500-price-to-sales/table/by-quarter'],
        [mult_sp500_shiller_pe_10_ratio, multpl, 'https://www.multpl.com/shiller-pe/table/by-month'],
        [mult_sp500_sales_per_share, multpl, 'https://www.multpl.com/s-p-500-sales/table/by-quarter'],
        [mult_sp500_sales_per_share_growth, multpl, 'https://www.multpl.com/s-p-500-sales-growth/table/by-quarter'],
        [mult_trate_1_month, multpl, 'https://www.multpl.com/1-month-treasury-rate/table/by-month'],
        [mult_trate_1_year, multpl, 'https://www.multpl.com/1-year-treasury-rate/table/by-month'],
        [mult_trate_10_year, multpl, 'https://www.multpl.com/10-year-treasury-rate/table/by-month'],
        [mult_trate_2_year, multpl, 'https://www.multpl.com/2-year-treasury-rate/table/by-month'],
        [mult_trate_20_year, multpl, 'https://www.multpl.com/20-year-treasury-rate/table/by-month'],
        [mult_trate_3_year, multpl, 'https://www.multpl.com/3-year-treasury-rate/table/by-month'],
        [mult_trate_30_year, multpl, 'https://www.multpl.com/30-year-treasury-rate/table/by-month'],
        [mult_trate_5_year, multpl, 'https://www.multpl.com/5-year-treasury-rate/table/by-month'],
        [mult_trate_6_month, multpl, 'https://www.multpl.com/6-month-treasury-rate/table/by-month'],
        # for computed ids the url is a list of dependencies (i.e. a list of data series ids)
        [sp500_div_reinvest_month, compute, [sp500_real_price_month, sp500_div_yield_month]],
        [sp500_earnings_growth, compute, [sp500_pe_ratio_month, sp500_real_price_month]],
        [sp500_earnings_yield, compute, [sp500_pe_ratio_month, sp500_real_price_month]]
    ]
