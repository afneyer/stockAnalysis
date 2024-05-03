
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
    cpi_urban_month = 'CPI_Urban_Month'
    ten_year_treasury_month = 'Ten_Year_Treasury'
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

    urls = [
        [sp500_pe_ratio_month, quandle, 'MULTPL/SP500_PE_RATIO_MONTH'],
        [sp500_div_yield_month, quandle, 'MULTPL/SP500_DIV_YIELD_MONTH'],
        [sp500_real_price_month, quandle, 'MULTPL/SP500_REAL_PRICE_MONTH'],
        [cpi_urban_month, multpl, 'https://www.multpl.com/cpi/table/by-month'],
        [ten_year_treasury_month, multpl, 'https://www.multpl.com/10-year-treasury-rate/table/by-month'],
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
        # for computed ids the url is a list of dependencies (i.e. a list of data series ids)
        [sp500_div_reinvest_month, compute, [sp500_real_price_month, sp500_div_yield_month]],
        [sp500_earnings_growth, compute, [sp500_pe_ratio_month, sp500_real_price_month]],
        [sp500_earnings_yield, compute, [sp500_pe_ratio_month, sp500_real_price_month]]
    ]
