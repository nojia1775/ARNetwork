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
		ARNetwork arn("test.json");
		// arn.randomize_weights(0, -sqrt(6 / 4), sqrt(6 / 4));
		// arn.randomize_weights(1, -sqrt(2), sqrt(2));
		// arn.randomize_bias(-0.1, 0.1);
		// arn.set_learning_rate(0.02);
		// std::vector<double> losses = arn.train(PairFunction("BCE"), PairFunction("tanh"), PairFunction("sigmoid"), arn.batching(inputs, 1), arn.batching(outputs, 1), 10000);
		arn.get_json("test1.json");
		// Vector<double> result;
		// result = arn.feed_forward(inputs[0], tanh, sigmoid);
		// std::cout << result[0] << std::endl;
		// result = arn.feed_forward(inputs[1], tanh, sigmoid);
		// std::cout << result[0] << std::endl;
		// result = arn.feed_forward(inputs[2], tanh, sigmoid);
		// std::cout << result[0] << std::endl;
		// result = arn.feed_forward(inputs[3], tanh, sigmoid);
		// std::cout << result[0] << std::endl;
	}
	catch (const std::exception& e) { std::cerr << e.what() << std::endl; }
	return 0;
}