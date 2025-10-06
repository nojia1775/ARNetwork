#include "../include/ARNetwork.hpp"

ARNetwork::ARNetwork(const size_t& inputs, const size_t& hidden_layers, const size_t& hidden_neurals, const size_t& outputs) : _inputs(inputs), _outputs(outputs), _weights(hidden_layers + 1), _z(hidden_layers + 1), _a(hidden_layers + 1), _bias(hidden_layers + 1), _learning_rate(0.1)
{
	for (size_t i = 0 ; i < hidden_layers + 1 ; i++)
	{
		if (i == 0)
		{
			_weights[i] = Matrix<double>(hidden_neurals ? hidden_neurals : 1, inputs);
			_bias[i] = Vector<double>(hidden_neurals ? hidden_neurals : 1);
		}
		else if (i == hidden_layers)
		{
			_weights[i] = Matrix<double>(outputs, hidden_neurals);
			_bias[i] = Vector<double>(outputs);
		}
		else
		{
			_weights[i] = Matrix<double>(hidden_neurals, hidden_neurals);
			_bias[i] = Vector<double>(hidden_neurals);
		}
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
	// std::cout << "FRONT\n";
	set_inputs(inputs);
	Matrix<double> neurals = _inputs;
	// std::cout << "INPUTS : \n";
	// neurals.display();
	// std::cout << std::endl;
	_a[0] = _inputs;
	for (size_t i = 0 ; i < nbr_hidden_layers() + 1 ; i++)
	{
		// std::cout << "WEIGHTS " << i << std::endl;
		// _weights[i].display();
		// std::cout << std::endl;
		// std::cout << "BIAS " << i << std::endl;
		// _bias[i].display();
		// std::cout << std::endl;
		_z[i] = _weights[i] * neurals + Matrix<double>(_bias[i]);
		// std::cout << "Z " << i << std::endl;
		// _z[i].display();
		// std::cout << std::endl;
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
		// std::cout << "A " << i << std::endl;
		// neurals.display();
		// std::cout << std::endl;
		if (i != nbr_hidden_layers())
			_a[i + 1] = neurals;
	}
	_outputs = Vector<double>(neurals);
	return _outputs;
}

void	ARNetwork::back_propagation(std::vector<Matrix<double>>& dW, std::vector<Matrix<double>>& dZ, const PairFunction& loss_functions, double (*d_layer_activation)(const double&), double (*d_output_activation)(const double&), const Vector<double>& y)
{
	// std::cout << "\nBACK\n";
	Matrix<double> dA(loss_functions.derived_foo(_outputs, y));
	for (int l = nbr_hidden_layers() ; l >= 0 ; l--)
	{
		// std::cout << "Z " << l << std::endl;
		// _z[l].display();
		// std::cout << std::endl;
		// std::cout << "Z apply " << l << std::endl;
		// _z[l].apply(l == (int)nbr_hidden_layers() ? d_output_activation : d_layer_activation).display();
		// std::cout << std::endl;
		// std::cout << "dA " << l << std::endl;
		// dA.display();
		// std::cout << std::endl;
		Matrix<double> z = dA.hadamard(_z[l].apply(l == (int)nbr_hidden_layers() ? d_output_activation : d_layer_activation));
		// std::cout << "A transpose " << l << std::endl;
		// Matrix<double>(_a[l]).transpose().display();
		// std::cout << std::endl;
		// std::cout << "dZ " << l << std::endl;
		// z.display();
		// std::cout << std::endl;
		Matrix<double> w = z * Matrix<double>(_a[l]).transpose();
		// std::cout << "dW " << l << std::endl;
		// w.display();
		// std::cout << std::endl;
		dZ[l] = dZ[l] + z;
		dW[l] = dW[l] + w;
		// std::cout << "WEIGHTS TRANSPOSE " << l << std::endl;
		// _weights[l].transpose().display();
		// std::cout << std::endl;
		dA = _weights[l].transpose() * z;
	}
}

void	ARNetwork::update_weights_bias(const std::vector<Matrix<double>>& dW, const std::vector<Matrix<double>>& dZ, const size_t& batch)
{
	// std::cout << "\nUPDATE\n";
	for (size_t layer = 0 ; layer < nbr_hidden_layers() + 1 ; layer++)
	{
		// std::cout << "CURRENT WEIGHTS " << layer << std::endl;
		// _weights[layer].display();
		// std::cout << "dW" << layer << std::endl;
		// dW[layer].display();
		// std::cout << "CURRENT BIAS " << layer << std::endl;
		// _bias[layer].display();
		// std::cout << "dZ " << layer << std::endl;
		// dZ[layer].display();
		// std::cout << "LEARNING RATE " << _learning_rate << std::endl;
		// std::cout << "1 / BATCH " << 1 / batch << std::endl;
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
	valid_lists(inputs, outputs, nbr_inputs(), nbr_outputs());
	std::vector<double> losses;
	for (size_t i = 0 ; i < epochs ; i++)
	{
		double loss_index = 0;
		std::vector<Matrix<double>> dW(nbr_hidden_layers() + 1);
		std::vector<Matrix<double>> dZ(nbr_hidden_layers() + 1);
		for (size_t j = 0 ; j < inputs.size() ; j++) // list of examples
		{
			for (size_t k = 0 ; k < inputs[j].size() ; k++) // batch of examples
			{
				Vector<double> prediction = feed_forward(inputs[j][k], layer_functions.get_activation_function(), output_functions.get_activation_function());
				loss_index = loss_functions.foo(prediction, outputs[j][k]) / prediction.dimension();
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
			_bias[i][j] = random_double(min, max);
		for (size_t j = 0 ; j < _weights[i].getNbrLines() ; j++)
		{
			for (size_t k = 0 ; k < _weights[i].getNbrColumns() ; k++)
				_weights[i][j][k] = random_double(min, max);
		}
	}
}