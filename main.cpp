#include "neural_network/include/ARNetwork.hpp"

int	main(void)
{
	std::vector<std::vector<double>> inputs({
		{1, 1},
		{1, 0},
		{0, 1},
		{0, 0}
	});
	std::vector<std::vector<double>> outputs({
		{0},
		{1},
		{1},
		{0}
	});
	try
	{
		ARNetwork arn(2, 1, 2, 1);
		std::vector<double> losses = arn.train(PairFunction("MSE"), PairFunction("sigmoid"), PairFunction("sigmoid"), arn.batching(inputs, 1), arn.batching(outputs, 1), 100);
		for (const auto& loss : losses)
			std::cout << loss << std::endl;
	}
	catch (const std::exception& e) { std::cerr << e.what() << std::endl; }
	return 0;
}