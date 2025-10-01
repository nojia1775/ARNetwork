#include "../include/ARNetwork.hpp"

ARNetwork::ARNetwork(const size_t& inputs, const size_t& hidden_layers, const size_t& hidden_neurals, const size_t& outputs) : _inputs(inputs), _outputs(outputs), _weights(hidden_layers + 1), _z(hidden_layers + 1), _bias(hidden_layers + 1), _learning_rate(0.1)
{
	for (size_t i = 0 ; i < hidden_layers + 1 ; i++)
	{
		_bias[i] = Vector<double>(hidden_neurals ? hidden_neurals : 1);
		if (i == 0)
			_weights[i] = Matrix<double>(hidden_neurals ? hidden_neurals : 1, inputs);
		else if (i == hidden_layers)
			_weights[i] = Matrix<double>(outputs, hidden_neurals);
		else
			_weights[i] = Matrix<double>(hidden_neurals, hidden_neurals);
		for (size_t j = 0 ; j < _weights[i].getNbrLines() ; j++)
			_bias[i][j] = random_double(-1, 1);
		for (size_t j = 0 ; j < _weights[i].getNbrLines() ; j++)
		{
			for (size_t k = 0 ; k < _weights[i].getNbrColumns() ; k++)
				_weights[i][j][k] = random_double(-1, 1);
		}
	}
}

ARNetwork::ARNetwork(const ARNetwork& arn) : _inputs(arn._inputs), _outputs(arn._outputs), _weights(arn._weights), _z(arn._z), _bias(arn._bias), _learning_rate(arn._learning_rate) {}

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
	for (size_t i = 0 ; i < nbr_hidden_layers() + 1 ; i++)
	{
		_z[i] = _weights[i] * neurals + Matrix<double>(_bias[i]);
		neurals = _z[i];
		if (i == nbr_hidden_layers())
		{
			if (output_activation)
				neurals.apply(output_activation);
		}
		else
		{
			if (layer_activation)
				neurals.apply(layer_activation);
		}
	}
	_outputs = Vector<double>(neurals);
	return _outputs;
}

double	ARNetwork::back_propagation(double (*loss)(const double&, const double&), double (*d_loss)(const double&, const double&), double (*d_layer_activation)(const double&), double (*d_output_activation)(const double&), const std::vector<double>& y)
{
	if (loss == NULL)
		throw Error("Error: loss function is missing");
	if (d_loss == NULL)
		throw Error("Error: derived loss function is missing");
	Matrix<double> dA(nbr_outputs(), 1);
	for (size_t i = 0 ; i < nbr_outputs() ; i++)
		dA[i][0] = d_loss(_outputs[i], y[i]);
	for (int l = nbr_hidden_layers() ; l >= 0 ; l--)
	{
		Matrix<double> dZ(dA * Matrix<double>(_z[l]).apply(d_output_activation));
	}
}