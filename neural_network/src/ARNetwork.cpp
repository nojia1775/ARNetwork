#include "../include/ARNetwork.hpp"

ARNetwork::ARNetwork(const size_t& inputs, const size_t& hidden_layers, const size_t& hidden_neurals, const size_t& outputs) : _inputs(inputs), _outputs(outputs), _weights(hidden_layers + 1), _z(hidden_layers + 1), _a(hidden_layers + 1), _bias(hidden_layers + 1), _learning_rate(0.1)
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

std::vector<double>	ARNetwork::feed_forward(const Vector<double>& inputs, double (*layer_activation)(const double&), double (*output_activation)(const double&))
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
				neurals.apply(output_activation);
		}
		else
		{
			if (layer_activation)
				neurals.apply(layer_activation);
		}
		if (i != nbr_hidden_layers())
			_a[i + 1] = neurals;
	}
	_outputs = Vector<double>(neurals);
	return _outputs.getStdVector();
}

void	ARNetwork::back_propagation(Matrix<double>& dW, Matrix<double>& dZ, double (*loss)(const double&, const double&), double (*d_loss)(const double&, const double&), double (*d_layer_activation)(const double&), double (*d_output_activation)(const double&), const std::vector<double>& y)
{
	if (loss == NULL)
		throw Error("Error: loss function is missing");
	if (d_loss == NULL)
		throw Error("Error: derived loss function is missing");
	Matrix<double> dA(nbr_outputs(), 1);
	double loss_index = 0;
	for (size_t i = 0 ; i < nbr_outputs() ; i++)
	{
		dA[i][0] = d_loss(_outputs[i], y[i]);
		loss_index += dA[i][0];
	}
	loss_index /= dA.getNbrLines();
	for (int l = nbr_hidden_layers() ; l >= 0 ; l--)
	{
		Matrix<double> tmp(_z[l]);
		tmp.apply(d_output_activation);
		Matrix<double> z(dA * tmp);
		Matrix<double> w(z * Matrix<double>(_a[l]).transpose());
		dZ = dZ + z;
		dW = dW + w;
	}
}

void	ARNetwork::update_weights_bias(const Matrix<double>& dW, const Matrix<double>& dZ, const size_t& layer, const size_t& batch)
{
	_weights[layer] = _weights[layer] - dW * _learning_rate * (1 / batch);
	_bias[layer] = Matrix<double>(_bias[layer]) - dZ * _learning_rate * (1 / batch);
}

std::vector<double>	ARNetwork::train(const size_t& batch, double (*loss)(const double&, const double&), double (*d_loss)(const double&, const double&), double (*layer_activation)(const double&), double (*d_layer_activation)(const double&), double (*output_activation)(const double&), double (*d_output_activation)(const double&), const std::vector<std::vector<std::vector<double>>>& inputs, const std::vector<std::vector<std::vector<double>>>& outputs, const size_t& epochs)
{
	Matrix<double> dW;
	Matrix<double> dZ;
	double loss_index = 0;
	std::vector<double> losses;
	for (size_t i = 0 ; i < epochs ; i++)
	{
		for (size_t j = 0 ; j < inputs.size() ; j++) // list of examples
		{
			for (size_t k = 0 ; k < inputs[0].size() ; k++) // batch of examples
			{
				std::vector<double> prediction = feed_forward(inputs[j][k], layer_activation, output_activation);
				for (size_t l = 0 ; l < prediction.size() ; l++)
					loss_index += loss(prediction[l], outputs[j][k][l]);
				back_propagation(dW, dZ, loss, d_loss, d_layer_activation, d_output_activation, prediction);
			}
			losses.push_back(loss_index / inputs[0].size());
			update_weights_bias(dW, dZ, j, batch);
		}
	}
	return losses;
}