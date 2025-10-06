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
		std::vector<double> losses = arn.train(PairFunction("MSE"), PairFunction("sigmoid"), PairFunction("sigmoid"), arn.batching(inputs, 1), arn.batching(outputs, 1), 1);
		// for (const auto& loss : losses)
		// 	std::cout << loss << std::endl;
		// std::vector<double> result;
		// result = arn.feed_forward(inputs[0], sigmoid, sigmoid);
		// std::cout << result[0] << std::endl;
		// result = arn.feed_forward(inputs[1], sigmoid, sigmoid);
		// std::cout << result[0] << std::endl;
		// result = arn.feed_forward(inputs[2], sigmoid, sigmoid);
		// std::cout << result[0] << std::endl;
		// result = arn.feed_forward(inputs[3], sigmoid, sigmoid);
		// std::cout << result[0] << std::endl;
	}
	catch (const std::exception& e) { std::cerr << e.what() << std::endl; }
	return 0;
}