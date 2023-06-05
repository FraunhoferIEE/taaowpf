import torch

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

import pytorch_lightning as pl

import pandas as pd


def collate_batch(batch):
    """Collate a batch of samples.

    This function filters out any samples that are None and then uses the default_collate function
    from torch.utils.data._utils.collate to collate the remaining samples.

    Args:
        batch: List of samples.

    Returns:
        Any: Collated batch of samples.
    """
    batch = filter(lambda sample: sample is not None, batch)
    return default_collate(list(batch))


def standardize(dataframe: pd.DataFrame, mean: float, std: float):
    """Standardizes an entire DataFrame using a given mean and standard deviation.

    This function subtracts the mean from each value in the DataFrame and then divides
    it by the standard deviation to standardize the values.

    Args:
        dataframe (pd.DataFrame): The DataFrame to be standardized.
        mean (float): The mean value used for standardization.
        std (float): The standard deviation value used for standardization.

    Returns:
        pd.DataFrame: The standardized DataFrame.
    """
    # Subtract the mean from each value and divide by the standard deviation
    dataframe_standardized = (dataframe - mean) / std
    return dataframe_standardized


class WPF_SingleTurbine_Dataset(Dataset):
    def __init__(
        self,
        windspeed: pd.DataFrame,
        windpower: pd.DataFrame,
        forecast_horizon: int = 8,
        n_past_timesteps: int = 0,
    ):
        """Dataset for wind power forecasting for individual wind farms.

        This dataset represents wind power forecasting data for a single wind farm. It takes wind speed
        and wind power measurements as inputs and provides samples for model training.

        Args:
            windspeed (pd.DataFrame): DataFrame containing windspeed data.
            windpower (pd.DataFrame): DataFrame containing windpower data.
            forecast_horizon (int, optional): Number of time steps to forecast. Defaults to 8.
            n_past_timesteps (int, optional): Number of past time steps to consider. Defaults to 0.
        """
        self.windpower = windpower.sort_index()
        self.windspeed = windspeed.sort_index()

        self.forecast_horizon = forecast_horizon
        self.n_past_timesteps = n_past_timesteps

        # wind speed for totally 'forecast_horizon' time steps [timesteps x features]
        self.shape_input_windspeed = torch.Size([self.forecast_horizon, 1])

        # wind power measurements for totally 'n_past_timesteps' time steps [timesteps x features]
        self.shape_input_windpower = torch.Size([self.n_past_timesteps, 1])

        # wind power measurements for the forecast horizon
        self.shape_target = torch.Size([self.forecast_horizon])

        self.shape = self.__getshape__()
        self.size = self.__getsize__()

    def __getitem__(self, index):
        """Get a sample from the dataset.

        Args:
            index (int): Index of the sample.

        Returns:
            Union[Tuple, None]: A tuple containing input_sample_windspeed, input_sample_windpower, and target_sample,
                or None if the sample does not meet the criteria.
        """
        index += self.n_past_timesteps
        forecast_start_time = self.windpower.iloc[index].name

        windspeed_forecast = self.get_windspeed_forecast(
            forecast_start_time, self.forecast_horizon, self.windspeed
        )
        input_sample_windspeed = torch.Tensor(
            windspeed_forecast["wind_speed_100m"].to_numpy()
        )
        input_sample_windspeed = torch.unsqueeze(input_sample_windspeed, dim=-1)

        if self.n_past_timesteps > 0:
            windpower_past = self.windpower.iloc[
                (index - self.n_past_timesteps) : index
            ]

            # replace nan values by linear interpolation if any
            windpower_past = windpower_past.interpolate(limit_direction="both")

            input_sample_windpower = torch.Tensor(windpower_past.to_numpy())

        else:
            input_sample_windpower = None

        # Replace nans by zeros since nan values might make problems
        # since data is standardized, we replace nan values by the mean of wind speed in the entire standardized dataset
        input_sample_windspeed = torch.nan_to_num(input_sample_windspeed, nan=0.0)

        # create targets from wind power
        target_sample = self.windpower.iloc[index : (index + self.forecast_horizon)]
        target_sample = torch.Tensor(target_sample.to_numpy())
        target_sample = torch.squeeze(target_sample, dim=1)

        # Only return samples with valid input and target shapes and a target without nan values
        if (
            (input_sample_windspeed.shape == self.shape_input_windspeed)
            & (input_sample_windpower.shape == self.shape_input_windpower)
            & (target_sample.shape == self.shape_target)
            & (torch.isnan(target_sample).any().item() == False)
        ):
            sample = (input_sample_windspeed, input_sample_windpower, target_sample)
        else:
            sample = None

        return sample

    def __len__(self):
        """Return the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.windpower) - self.forecast_horizon - self.n_past_timesteps

    def __getshape__(self):
        """Return the shape information of the dataset.

        The shape information includes the length of the dataset,
        the shape of the windspeed input, the shape of the windpower input,
        and the shape of the target.

        Returns:
            Tuple[int, torch.Size, torch.Size, torch.Size]: Shape information of the dataset.
        """
        return (
            self.__len__(),
            self.shape_input_windspeed,
            self.shape_input_windpower,
            self.shape_target,
        )

    def __getsize__(self):
        """Return the size of the dataset.

        Returns:
            int: Size of the dataset.
        """
        return self.__len__()

    def get_windspeed_forecast(self, forecast_start_time, forecast_horizon, windspeed):
        """Get the windspeed forecast for the given start time and horizon.

        Args:
            forecast_start_time (pd.Timestamp): The start time of the forecast.
            forecast_horizon (int): The number of hours in the forecast horizon.
            windspeed (pd.DataFrame): The windspeed data.

        Returns:
            pd.DataFrame: The windspeed forecast.
        """
        forecast_times = [
            (forecast_start_time + pd.Timedelta(hours=hour))
            for hour in range(0, forecast_horizon)
        ]

        windspeed_forecast = windspeed.reindex(index=forecast_times)

        return windspeed_forecast


class WPF_SingleTurbine_DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        forecast_horizon: int = 8,
        n_past_timesteps: int = 0,
        batch_size: int = 256,
        num_workers: int = 32,
    ):
        """Data module for the WPF Single Turbine dataset.

        Args:
            data_dir (str): The directory path to the dataset.
            forecast_horizon (int, optional): The number of hours in the forecast horizon. Defaults to 8.
            n_past_timesteps (int, optional): The number of past timesteps to consider. Defaults to 0.
            batch_size (int, optional): The batch size. Defaults to 256.
            num_workers (int, optional): The number of workers for data loading. Defaults to 32.
        """
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.forecast_horizon = forecast_horizon
        self.n_past_timesteps = n_past_timesteps

        self.data_dir = data_dir

    def setup(self, stage=None):
        """Prepare the dataset.

        Args:
            stage (optional): The stage of setup. Defaults to None.
        """
        # Input data and targets
        ## Load wind speed and wind power data
        data = pd.read_csv(
            self.data_dir,
            parse_dates=True,
            infer_datetime_format=True,
            index_col="TIMESTAMP",
        )

        windpower = pd.DataFrame(data["wind_power"])

        windspeed = pd.DataFrame(data["wind_speed_100m"])

        ## Select training, validation and test data
        ### wind power data
        self.windpower_train = pd.concat(
            [
                windpower.loc["2012-01-01":"2012-03-24"].copy(),
                windpower.loc["2012-04-01":"2012-06-23"].copy(),
                windpower.loc["2012-07-01":"2012-09-23"].copy(),
                windpower.loc["2012-10-01":"2012-12-24"].copy(),
                windpower.loc["2013-01-01":"2013-03-24"].copy(),
                windpower.loc["2013-04-01":"2013-06-23"].copy(),
            ]
        )

        self.windpower_valid = pd.concat(
            [
                windpower.loc["2012-03-25":"2012-03-31"].copy(),
                windpower.loc["2012-06-24":"2012-06-30"].copy(),
                windpower.loc["2012-09-24":"2012-09-30"].copy(),
                windpower.loc["2012-12-25":"2012-12-31"].copy(),
                windpower.loc["2013-03-25":"2013-03-31"].copy(),
                windpower.loc["2013-06-24":"2013-06-30"].copy(),
            ]
        )

        self.windpower_test = windpower.loc["2013-07-01":"2013-12-31"].copy()

        ### wind speed data
        self.windspeed_train = windspeed.loc[self.windpower_train.index].copy()
        self.windspeed_valid = windspeed.loc[self.windpower_valid.index].copy()
        self.windspeed_test = windspeed.loc[self.windpower_test.index].copy()

        ## Standardize wind speed data
        ### Calculate mean and standard deviation based on training data (for the entire dataframe)
        self.mean_windspeed = self.windspeed_train["wind_speed_100m"].mean()
        self.std_windspeed = self.windspeed_train["wind_speed_100m"].std()

        self.windspeed_train["wind_speed_100m"] = standardize(
            self.windspeed_train["wind_speed_100m"],
            self.mean_windspeed,
            self.std_windspeed,
        )
        self.windspeed_valid["wind_speed_100m"] = standardize(
            self.windspeed_valid["wind_speed_100m"],
            self.mean_windspeed,
            self.std_windspeed,
        )
        self.windspeed_test["wind_speed_100m"] = standardize(
            self.windspeed_test["wind_speed_100m"],
            self.mean_windspeed,
            self.std_windspeed,
        )

        # Create datasets for training, validation and testing
        self.train_dataset = WPF_SingleTurbine_Dataset(
            windspeed=self.windspeed_train,
            windpower=self.windpower_train,
            forecast_horizon=self.forecast_horizon,
            n_past_timesteps=self.n_past_timesteps,
        )
        self.val_dataset = WPF_SingleTurbine_Dataset(
            windspeed=self.windspeed_valid,
            windpower=self.windpower_valid,
            forecast_horizon=self.forecast_horizon,
            n_past_timesteps=self.n_past_timesteps,
        )
        self.test_dataset = WPF_SingleTurbine_Dataset(
            windspeed=self.windspeed_test,
            windpower=self.windpower_test,
            forecast_horizon=self.forecast_horizon,
            n_past_timesteps=self.n_past_timesteps,
        )

    def train_dataloader(self):
        """Create a data loader for training.

        Returns:
            DataLoader: A DataLoader object for training data.
        """
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_batch,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return train_loader

    def val_dataloader(self):
        """Create a data loader for validation.

        Returns:
            DataLoader: A DataLoader object for validation data.
        """
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_batch,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return val_loader

    def test_dataloader(self):
        """Create a data loader for testing.

        Returns:
            DataLoader: A DataLoader object for test data.
        """
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_batch,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
        )
        return test_loader
