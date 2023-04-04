class Portfolio:

    def __init__(self, tickerString: str, expectedReturn: float, portfolioName: str, riskBucket: int):

        self.name = portfolioName
        self.riskBucket = riskBucket
        self.expectedReturn = expectedReturn
        self.allocations = []

        from pypfopt.efficient_frontier import EfficientFrontier
        from pypfopt import risk_models
        from pypfopt import expected_returns
    
        df = self.__getDailyPrices(tickerString, "20y")

        self.mu = expected_returns.mean_historical_return(df)
        self.S = risk_models.sample_cov(df)

        ef = EfficientFrontier(self.mu, self.S)

        ef.efficient_return(expectedReturn)
        self.expectedRisk = ef.portfolio_performance()[1]
        portfolioWeights = ef.clean_weights()

        for key, value in portfolioWeights.items():
            newAllocation = Allocation(key, value)
            self.allocations.append(newAllocation)

    def __getDailyPrices(self, tickerStringList, period):
        data = yf.download(tickerStringList, group_by="Ticker", period=period)
        data = data.iloc[:, data.columns.get_level_values(1)=="Close"]
        data = data.dropna()
        data.columns = data.columns.droplevel(1)
        return data

    def printPortfolio(self):
        print("Portfolio Name: " + self.name)
        print("Risk Bucket: " + str(self.riskBucket))
        print("Expected Return: " + str(self.expectedReturn))
        print("Expected Risk: " + str(self.expectedRisk))
        print("Allocations: ")
        for allocation in self.allocations:
            print("Ticker: " + allocation.ticker + ", Percentage: " + str(allocation.percentage))

    def showEfficientFrontier(self):
        import copy
        import numpy as np
        ef = EfficientFrontier(self.mu, self.S)
        fig, ax = pyplot.subplots()
        #ef_max_sharpe = copy.deepcopy(ef)
        ef_max_sharpe = EfficientFrontier(self.mu, self.S)
        #ef_return = copy.deepcopy(ef)
        ef_return = EfficientFrontier(self.mu, self.S)
        plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)

        # Generate random portfolios
        n_samples = 10000
        w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
        rets = w.dot(ef.expected_returns)
        stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
        sharpes = rets / stds
        ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

        # Find the tangency portfolio
        ef_max_sharpe.max_sharpe()
        ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
        ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")

        # Find the return portfolio
        ef_return.efficient_return(self.expectedReturn)
        ret_tangent2, std_tangent2, _ = ef_return.portfolio_performance()
        returnP = str(int(self.expectedReturn*100))+"%"
        ax.scatter(std_tangent2, ret_tangent2, marker="*", s=100, c="y", label=returnP)

        # Output
        ax.set_title("Efficient Frontier for " + returnP + " returns")
        ax.legend()
        pyplot.figure(figsize=(30,20))
        pyplot.tight_layout()
        pyplot.show()
