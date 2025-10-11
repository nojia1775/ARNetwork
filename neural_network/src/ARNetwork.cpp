#include "../include/ARNetwork.hpp"

ARNetwork::ARNetwork(const std::vector<size_t>& network)
{
	if (network.size() < 2)
		throw Error("Error: not enough neurals in the network");
	for (const auto& neurals : network)
		if (neurals == 0)
			throw Error("Error: number of neurals can't be 0");
	size_t inputs = network[0];
	size_t outputs = network[network.size() - 1];
	size_t hidden_layers = network.size() - 2;
	_weights = std::vector<Matrix<double>>(hidden_layers + 1);
	_bias = std::vector<Vector<double>>(hidden_layers + 1);
	_inputs = Vector<double>(inputs);
	_outputs = Vector<double>(outputs);
	_z = std::vector<Vector<double>>(hidden_layers + 1);
	_a = std::vector<Vector<double>>(hidden_layers + 1);
	_learning_rate = 0.1;
	for (size_t i = 0 ; i < hidden_layers + 1 ; i++)
	{
		_weights[i] = Matrix<double>(network[i + 1], network[i]);
		_bias[i] = Vector<double>(network[i + 1]);
		for (size_t j = 0 ; j < _weights[i].getNbrLines() ; j++)
			_bias[i][j] = random_double(-1, 1);
		for (size_t j = 0 ; j < _weights[i].getNbrLines() ; j++)
		{
			for (size_t k = 0 ; k < _weights[i].getNbrColumns() ; k++)
				_weights[i][j][k] = random_double(-1, 1);
		}
	}
}

ARNetwork::ARNetwork(const ARNetwork& arn) : _inputs(arn._inputs), _outputs(arn._outputs), _weights(arn._weights), _z(arn._z), _a(arn._a), _bias(arn._bias), _learning_rate(arn._learning_rate) {}

ARNetwork	ARNetwork::operator=(const ARNetwork& arn)
{
	if (this != &arn)
	{
		_inputs = arn._inputs;
		_outputs = arn._outputs;
		_weights = arn._weights;
		_z = arn._z;
		_a = arn._a;
		_bias = arn._bias;
		_learning_rate = arn._learning_rate;
		_hidden_activation = arn._hidden_activation;
		_output_activation = arn._output_activation;
		_loss = arn._loss;
	}
	return *this;
}

const double&	ARNetwork::get_bias(const size_t& i, const size_t& j) const
{
	if (i > _bias.size() - 1)
		throw Error("Error: index out of range");
	if (j < _bias[i].dimension() - 1)
		throw Error("Error: index out of range");
	return _bias[i][j];
}

void	ARNetwork::set_bias(const size_t& i, const size_t& j, const double& bias)
{
	if (i > _bias.size() - 1)
		throw Error("Error: index out of range");
	if (j < _bias[i].dimension() - 1)
		throw Error("Error: index out of range");
	_bias[i][j] = bias;
}

Vector<double>	ARNetwork::feed_forward(const Vector<double>& inputs, double (*layer_activation)(const double&), double (*output_activation)(const double&))
{
	set_inputs(inputs);
	Matrix<double> neurals = _inputs;
	_a[0] = _inputs;
	for (size_t i = 0 ; i < nbr_hidden_layers() + 1 ; i++)
	{
		_z[i] = _weights[i] * neurals + Matrix<double>(_bias[i]);
		neurals = _z[i];
		if (i == nbr_hidden_layers())
		{
			if (output_activation)
				neurals = neurals.apply(output_activation);
		}
		else
		{
			if (layer_activation)
				neurals = neurals.apply(layer_activation);
		}
		if (i != nbr_hidden_layers())
			_a[i + 1] = neurals;
	}
	_outputs = Vector<double>(neurals);
	return _outputs;
}

void	ARNetwork::back_propagation(std::vector<Matrix<double>>& dW, std::vector<Matrix<double>>& dZ, const PairFunction& loss_functions, double (*d_layer_activation)(const double&), double (*d_output_activation)(const double&), const Vector<double>& y)
{
	Matrix<double> dA(loss_functions.derived_foo(_outputs, y));
	// dA.display();
	// double loss = dA.sumLines()[0][0] / dA.getNbrLines();
	for (int l = nbr_hidden_layers() ; l >= 0 ; l--)
	{
		Matrix<double> z = dA.hadamard(_z[l].apply(l == (int)nbr_hidden_layers() ? d_output_activation : d_layer_activation));
		Matrix<double> w = z * Matrix<double>(_a[l]).transpose();
		dZ[l] = dZ[l] + z;
		dW[l] = dW[l] + w;
		dA = _weights[l].transpose() * z;
	}
	// return loss;
}

void	ARNetwork::update_weights_bias(const std::vector<Matrix<double>>& dW, const std::vector<Matrix<double>>& dZ, const size_t& batch)
{
	for (size_t layer = 0 ; layer < nbr_hidden_layers() + 1 ; layer++)
	{
		_weights[layer] = _weights[layer] - dW[layer] * _learning_rate * static_cast<double>(1.0 / static_cast<double>(batch));
		_bias[layer] = Matrix<double>(_bias[layer]) - dZ[layer] * _learning_rate * static_cast<double>(1.0 / static_cast<double>(batch));
	}
}

static void	valid_lists(const std::vector<std::vector<std::vector<double>>>& inputs, const std::vector<std::vector<std::vector<double>>>& outputs, const size_t& nbr_inputs, const size_t& nbr_outputs)
{
	if (inputs.size() != outputs.size())
		throw Error("Error: the number of batch of inputs and outputs must be the same");
	for (size_t i = 0 ; i < inputs.size() ; i++)
	{
		if (inputs[i].size() != outputs[i].size())
			throw Error("Error: batch of inputs and outputs have different size");
		for (size_t j = 0 ; j < inputs[i].size() ; j++)
		{
			if (inputs[i][j].size() != nbr_inputs)
				throw Error("Error: example " + j + std::string(" must have " + nbr_inputs + std::string(" inputs")));
			if (outputs[i][j].size() != nbr_outputs)
				throw Error("Error: example " + j + std::string(" must have " + nbr_inputs + std::string(" outputs")));
		}
	}
}

std::vector<double>	ARNetwork::train(const PairFunction& loss_functions, const PairFunction& layer_functions, const PairFunction& output_functions, const std::vector<std::vector<std::vector<double>>>& inputs, const std::vector<std::vector<std::vector<double>>>& outputs, const size_t& epochs)
{
	if (inputs.empty())
		throw Error("Error: there is no input");
	if (outputs.empty())
		throw Error("Error: there is no expected output");
	_loss = loss_functions.get_loss_name();
	_hidden_activation = layer_functions.get_activation_name();
	_output_activation = output_functions.get_activation_name();
	valid_lists(inputs, outputs, nbr_inputs(), nbr_outputs());
	std::vector<double> losses;
	for (size_t i = 0 ; i < epochs ; i++)
	{
		double loss_index = 0;
		for (size_t j = 0 ; j < inputs.size() ; j++)
		{
			std::vector<Matrix<double>> dW(nbr_hidden_layers() + 1);
			std::vector<Matrix<double>> dZ(nbr_hidden_layers() + 1);
			for (size_t k = 0 ; k < inputs[j].size() ; k++)
			{
				Vector<double> prediction = feed_forward(inputs[j][k], layer_functions.get_activation_function(), output_functions.get_activation_function());
				loss_index += loss_functions.foo(prediction, outputs[j][k]);
				back_propagation(dW, dZ, loss_functions, layer_functions.get_derived_activation_function(), output_functions.get_derived_activation_function(), outputs[j][k]);
			}
			losses.push_back(loss_index / inputs[j].size());
			update_weights_bias(dW, dZ, inputs[j].size());
		}
	}
	return losses;
}

std::vector<std::vector<std::vector<double>>>	ARNetwork::batching(const std::vector<std::vector<double>>& list, const size_t& batch)
{
	if (batch == 0)
		throw Error("Error: batch cannot be 0");
	size_t groups = batch > list.size() ? 1 : (size_t)(list.size() / batch);
	groups += list.size() % batch == 0 ? 0 : 1;
	std::vector<std::vector<std::vector<double>>> result(groups);
	size_t index = 0;
	for (size_t i = 0 ; i < list.size() ; i++)
	{
		if (i != 0 && i % batch == 0)
			index++;
		result[index].push_back(list[i]);
	}
	return result;
}

void	ARNetwork::randomize_weights(const double& min, const double& max)
{
	for (size_t i = 0 ; i < nbr_hidden_layers() + 1 ; i++)
	{
		for (size_t j = 0 ; j < _weights[i].getNbrLines() ; j++)
		{
			for (size_t k = 0 ; k < _weights[i].getNbrColumns() ; k++)
				_weights[i][j][k] = random_double(min, max);
		}
	}
}

void	ARNetwork::randomize_weights(const size_t& layer, const double& min, const double& max)
{
	if (layer > nbr_hidden_layers())
		throw Error("Error: index out of range");
	for (size_t i = 0 ; i < _weights[layer].getNbrLines() ; i++)
	{
		for (size_t j = 0 ; j < _weights[layer].getNbrColumns() ; j++)
			_weights[layer][i][j] = random_double(min, max);
	}
}

void	ARNetwork::randomize_bias(const double& min, const double& max)
{
	size_t i;
	for (i = 0 ; i < nbr_hidden_layers() ; i++)
	{
		for (size_t j = 0 ; j < nbr_hidden_neurals(i) ; j++)
			_bias[i][j] = random_double(min, max);
	}
	for (size_t j = 0 ; j < nbr_outputs() ; j++)
		_bias[i][j] = random_double(min, max);
}

void	ARNetwork::randomize_bias(const size_t& layer, const double& min, const double& max)
{
	if (layer > nbr_hidden_layers())
		throw Error("Error: index out of range");
	for (size_t j = 0 ; j < nbr_hidden_neurals(layer) ; j++)
		_bias[layer][j] = random_double(min, max);
}