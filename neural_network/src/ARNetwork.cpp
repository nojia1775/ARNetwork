#include "../include/ARNetwork.hpp"

ARNetwork::ARNetwork(const size_t& inputs, const size_t& hidden_layers, const size_t& hidden_neurals, const size_t& outputs) : _inputs(inputs), _outputs(outputs), _weights(hidden_layers + 1), _bias(hidden_layers + 1), _learning_rate(0.1)
{
	for (size_t i = 0 ; i < hidden_layers + 1 ; i++)
	{
		if (i == 0)
			_weights[i] = Matrix<double>(hidden_neurals, inputs);
		else if (i == hidden_layers)
			_weights[i] = Matrix<double>(outputs, hidden_neurals);
		else
			_weights[i] = Matrix<double>(hidden_neurals, hidden_neurals);
		for (size_t j = 0 ; j < _weights[i].getNbrLines() ; j++)
		{
			for (size_t k = 0 ; k < _weights[i].getNbrColumns() ; k++)
				_weights[i][j][k] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) * 2 - 1;
		}
	}
	for (auto& bias : _bias)
		bias = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) * 2 - 1;
}

ARNetwork::ARNetwork(const ARNetwork& arn) : _inputs(arn._inputs), _outputs(arn._outputs), _weights(arn._weights), _bias(arn._bias), _learning_rate(arn._learning_rate) {}

ARNetwork	ARNetwork::operator=(const ARNetwork& arn)
{
	if (this != &arn)
	{
		_inputs = arn._inputs;
		_outputs = arn._outputs;
		_weights = arn._weights;
		_bias = arn._bias;
		_learning_rate = arn._learning_rate;
	}
	return *this;
}