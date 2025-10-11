#include "../include/ARNetwork.hpp"

std::vector<double>	softMax(const std::vector<double>& input)
{
	std::vector<double> output(input.size());
	double maxVal = *std::max_element(input.begin(), input.end());
	double sumExp = 0.0;
	for (size_t i = 0; i < input.size(); ++i)
	{
		output[i] = std::exp(input[i] - maxVal);
		sumExp += output[i];
	}
	for (size_t i = 0; i < output.size(); ++i)
		output[i] /= sumExp;
	return output;
}

std::vector<double>	derived_softMax(const std::vector<double>& inputs)
{
	std::vector<double> result(inputs.size());

	for (size_t i = 0 ; i < inputs.size() ; i++)
	{
		double sum = 0;
		for (size_t j = 0 ; j < inputs.size() ; j++)
			sum += exp(inputs[j]);
		result[i] = exp(inputs[i]) / sum;
	}
	return result;
}

double	crossEntropy(const std::vector<double>& yPred, const std::vector<double>& yTrue)
{
	double loss = 0.0f;
	for (size_t i = 0; i < yTrue.size(); ++i)
		loss -= std::log(yPred[i] + std::numeric_limits<double>::epsilon());
	return loss;
}

double	derived_crossEntropy(const std::vector<double>& yPred, const std::vector<double>& yTrue)
{
	double loss = 0;
	for (size_t i = 0 ; i < yPred.size() ; i++)
		loss += yTrue[i] * std::log(yPred[i] + std::numeric_limits<double>::epsilon());
	return -loss;
}

double	MSE(const Vector<double>& y_pred, const Vector<double>& y)
{
	if (y_pred.empty() || y.empty())
		throw Error("Error: vector is empty");
	if (y_pred.dimension() != y.dimension())
		throw Error("Error: vectors must have the same dimension");
	double sum = 0;
	for (size_t i = 0 ; i < y_pred.dimension() ; i++)
		sum += pow(y_pred[i] - y[i], 2);
	return sum / (2.0 * static_cast<double>(y_pred.dimension()));
}

Vector<double>	derived_MSE(const Vector<double>& y_pred, const Vector<double>& y)
{
	if (y_pred.empty() || y.empty())
		throw Error("Error: vector is empty");
	if (y_pred.dimension() != y.dimension())
		throw Error("Error: vectors must have the same dimension");
	Vector<double> gradients(y_pred.dimension());
	for (size_t i = 0 ; i < gradients.dimension() ; i++)
		gradients[i] = (y_pred[i] - y[i]) / y_pred.dimension();
	return gradients;
}

double	BCE(const Vector<double>& y_pred, const Vector<double>& y)
{
	if (y_pred.empty() || y.empty())
		throw Error("Error: vector is empty");
	if (y_pred.dimension() != y.dimension())
		throw Error("Error: vectors must have the same dimension");
	double sum = 0;
	for (size_t i = 0 ; i < y.dimension() ; i++)
	{
		double y_hat;
		if (y_pred[i] < std::numeric_limits<double>::epsilon())
			y_hat = std::numeric_limits<double>::epsilon();
		else if (y_pred[i] > 1 - std::numeric_limits<double>::epsilon())
			y_hat = 1 - std::numeric_limits<double>::epsilon();
		else 
			y_hat = y_pred[i];
		sum += y[i] * std::log(y_hat) + (1 - y[i]) * std::log(1 - y_hat);
	}
	return -sum / static_cast<double>(y_pred.dimension());
}

Vector<double>	derived_BCE(const Vector<double>& y_pred, const Vector<double>& y)
{
	if (y_pred.empty() || y.empty())
		throw Error("Error: vector is empty");
	if (y_pred.dimension() != y.dimension())
		throw Error("Error: vectors must have the same dimension");
	Vector<double> gradients(y_pred.dimension());
	for (size_t i = 0 ; i < y_pred.dimension() ; i++)
	{
		double y_hat;
		if (y_pred[i] < std::numeric_limits<double>::epsilon())
			y_hat = std::numeric_limits<double>::epsilon();
		else if (y_pred[i] > 1 - std::numeric_limits<double>::epsilon())
			y_hat = 1 - std::numeric_limits<double>::epsilon();
		else 
			y_hat = y_pred[i];
		gradients[i] = (y_hat - y[i]) / (y_hat * (1.0 - y_hat) * y_pred.dimension());
	}
	return gradients;
}