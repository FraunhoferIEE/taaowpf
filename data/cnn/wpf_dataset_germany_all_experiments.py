import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

import pytorch_lightning as pl

import pandas as pd
import numpy as np


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


class WPF_Germany_Dataset(Dataset):
    def __init__(
        self,
        windspeed: pd.DataFrame,
        windpower: pd.DataFrame,
        forecast_horizon: int = 8,
        n_past_timesteps: int = 0,
    ):
        """Dataset for wind power forecasting in Germany.

        This dataset represents wind power forecasting data in Germany. It takes wind speed
        and wind power measurements as inputs and provides samples for model training.

        Args:
            windspeed (pd.DataFrame): DataFrame containing wind speed measurements.
            windpower (pd.DataFrame): DataFrame containing wind power measurements.
            forecast_horizon (int): The number of future time steps to predict.
            n_past_timesteps (int): The number of past time steps to include in the input.

        """
        self.windpower = windpower.sort_index()
        self.windspeed = windspeed

        self.forecast_horizon = forecast_horizon
        self.n_past_timesteps = n_past_timesteps

        # wind speed for totally 'n_past_timesteps + forecast_horizon' time steps on a 100 x 85 grid [channels × pixel height × pixel width]
        self.shape_input = torch.Size(
            [(self.n_past_timesteps + self.forecast_horizon), 100, 85]
        )
        # wind power measurements for the forecast horizon
        self.shape_target = torch.Size([self.forecast_horizon])

        self.shape = self.__getshape__()
        self.size = self.__getsize__()

    def __getitem__(self, index):
        """Returns a sample from the dataset at the given index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the input sample and target sample.
        """
        index += self.n_past_timesteps
        forecast_start_time = self.windpower.iloc[index].name

        windspeed_forecast = self.get_windspeed_forecast(
            forecast_start_time, self.forecast_horizon, self.windspeed
        )

        if self.n_past_timesteps > 0:
            windspeed_past = self.get_windspeed_past(
                forecast_start_time, self.n_past_timesteps, self.windspeed
            )

            # concatenate future and past wind speed forecasts along time axis
            input_sample = np.concatenate((windspeed_past, windspeed_forecast), axis=0)

        else:
            input_sample = windspeed_forecast

        input_sample = torch.Tensor(input_sample)

        # Replace nans by zeros since nan values might make problems
        # since data is standardized, we replace nan values by the mean of wind speed in the entire standardized dataset
        input_sample = torch.nan_to_num(input_sample, nan=0.0)

        # create targets from wind power
        target_sample = self.windpower.iloc[index : (index + self.forecast_horizon)]
        target_sample = torch.Tensor(target_sample.to_numpy())
        target_sample = torch.squeeze(target_sample, dim=1)

        # Only return samples with valid input and target shapes and a target without nan values
        if (
            (input_sample.shape == self.shape_input)
            & (target_sample.shape == self.shape_target)
            & (torch.isnan(target_sample).any().item() == False)
        ):
            sample = (input_sample, target_sample)
        else:
            sample = None

        return sample

    def __len__(self):
        """Get the length of the WPF_Germany_Dataset dataset.

        Returns:
            int: The length of the WPF_Germany_Dataset dataset.
        """
        return len(self.windpower) - self.forecast_horizon - self.n_past_timesteps

    def __getshape__(self):
        """Get the shape of the WPF_Germany_Dataset dataset.

        Returns:
            Tuple[int, int, int]: A tuple representing the shape of the WPF_Germany_Dataset dataset.
                                 The tuple contains three integers: length, shape_input, and shape_target.
        """
        return (self.__len__(), self.shape_input, self.shape_target)

    def __getsize__(self):
        """Get the size of the WPF_Germany_Dataset dataset.

        Returns:
            int: The size of the WPF_Germany_Dataset dataset.
        """
        return self.__len__()

    def get_windspeed_forecast(self, forecast_start_time, forecast_horizon, windspeed):
        """Get the windspeed forecast for a given start time and horizon.

        Args:
            forecast_start_time (datetime): The start time of the forecast.
            forecast_horizon (int): The duration of the forecast.
            windspeed (pd.DataFrame): DataFrame containing the windspeed data.

        Returns:
            ndarray: The windspeed forecast as a numpy array.

        """
        forecast_date = pd.Timestamp(forecast_start_time.date())
        forecast_start_hour = forecast_start_time.hour
        forecast_times = [
            (forecast_start_time + pd.Timedelta(hours=hour))
            for hour in range(0, forecast_horizon)
        ]

        if (forecast_start_hour >= 5) and (
            forecast_start_hour < 17
        ):  # modelrun 0 is available at 4 o'clock
            modelrun = 0
        elif forecast_start_hour >= 17:  # modelrun 12 is available at 16 o'clock
            modelrun = 12
        elif forecast_start_hour < 5:
            modelrun = 12
            forecast_date = forecast_date - pd.Timedelta(days=1)

        modelrun_time = forecast_date + pd.Timedelta(hours=modelrun)
        modelrun_time = modelrun_time.tz_localize("UTC")

        windspeed_forecast = windspeed.loc[
            (windspeed.index.get_level_values("model_run [UTC]") == modelrun_time)
            & (
                windspeed.index.get_level_values("prediction_time [UTC]").isin(
                    forecast_times
                )
            )
        ]

        num_timesteps = windspeed_forecast.shape[0]

        # array of shape (H, W, C)
        windspeed_forecast = (
            windspeed_forecast.to_numpy().reshape(num_timesteps, 85, 100).T
        )

        # array of shape (C, H, W)
        windspeed_forecast = np.moveaxis(windspeed_forecast, -1, 0)

        return windspeed_forecast

    def get_windspeed_past(self, forecast_start_time, n_past_timesteps, windspeed):
        """Get the past windspeed data for a given forecast start time and number of past timesteps.

        Args:
            forecast_start_time (datetime): The start time of the forecast.
            n_past_timesteps (int): The number of past timesteps to retrieve.
            windspeed (pd.DataFrame): DataFrame containing the windspeed data.

        Returns:
            ndarray: The past windspeed data as a numpy array.

        """
        forecast_date = pd.Timestamp(forecast_start_time.date())
        forecast_start_hour = forecast_start_time.hour
        past_times = [
            (forecast_start_time - pd.Timedelta(hours=hour))
            for hour in reversed(range(1, n_past_timesteps + 1))
        ]

        if (forecast_start_hour >= 5) and (
            forecast_start_hour < 17
        ):  # modelrun 0 is available at 4 o'clock
            current_modelrun = 0
        elif forecast_start_hour >= 17:  # modelrun 12 is available at 16 o'clock
            current_modelrun = 12
        elif forecast_start_hour < 5:
            current_modelrun = 12
            forecast_date = forecast_date - pd.Timedelta(days=1)

        windspeed_past = None

        for past_time in past_times:
            time_diff = int((forecast_start_time - past_time) / np.timedelta64(1, "h"))
            if time_diff >= 5:
                past_date = pd.Timestamp(past_time.date())
                past_hour = past_time.hour

                if (past_hour >= 0) and (past_hour < 12):
                    modelrun = 0
                elif past_hour >= 12:
                    modelrun = 12

            else:
                modelrun = current_modelrun
                past_date = forecast_date

            modelrun_time = forecast_date + pd.Timedelta(hours=modelrun)
            modelrun_time = modelrun_time.tz_localize("UTC")

            windspeed_past_time = windspeed.loc[
                (windspeed.index.get_level_values("model_run [UTC]") == modelrun_time)
                & (
                    windspeed.index.get_level_values("prediction_time [UTC]")
                    == past_time
                )
            ]

            windspeed_past = pd.concat([windspeed_past, windspeed_past_time], axis=0)

        num_timesteps = windspeed_past.shape[0]

        # array of shape (H, W, C)
        windspeed_past = windspeed_past.to_numpy().reshape(num_timesteps, 85, 100).T

        # array of shape (C, H, W)
        windspeed_past = np.moveaxis(windspeed_past, -1, 0)

        return windspeed_past


class WPF_Germany_DataModule(pl.LightningDataModule):
    def __init__(
        self,
        windspeed_dir: str,
        windpower_dir: str,
        forecast_horizon: int = 8,
        n_past_timesteps: int = 0,
        batch_size: int = 256,
        num_workers: int = 32,
        experiment: int = 1,
    ):
        """
        Data module for WPF Germany dataset.

        Args:
            windspeed_dir (str): Path to the windspeed data file.
            windpower_dir (str): Path to the windpower data file.
            forecast_horizon (int, optional): Forecast horizon. Defaults to 8.
            n_past_timesteps (int, optional): Number of past timesteps. Defaults to 0.
            batch_size (int, optional): Batch size. Defaults to 256.
            num_workers (int, optional): Number of workers for data loading. Defaults to 32.
            experiment (int, optional): Experiment number. Defaults to 1.
        """
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.forecast_horizon = forecast_horizon
        self.n_past_timesteps = n_past_timesteps

        self.windpower_dir = windpower_dir
        self.windspeed_dir = windspeed_dir

        self.experiment = experiment

    def setup(self, stage=None):
        """
        Set up the data module.

        Args:
            stage (optional): Stage of setup. Defaults to None.
        """
        # Input data and targets
        ## Load wind speed and wind power data
        windspeed = pd.read_csv(
            self.windspeed_dir,
            index_col=["model_run [UTC]", "prediction_time [UTC]"],
            parse_dates=["model_run [UTC]", "prediction_time [UTC]"],
        )

        windpower = pd.read_csv(
            self.windpower_dir,
            index_col=["start_time_utc"],
            parse_dates=["start_time_utc"],
        )

        ## Select training, validation and test data
        ### wind power data

        if self.experiment == 1:
            self.windpower_train = pd.concat(
                [
                    windpower.loc["2019-01-01":"2019-01-27"].copy(),
                    windpower.loc["2019-02-01":"2019-02-24"].copy(),
                    windpower.loc["2019-03-01":"2019-03-27"].copy(),
                    windpower.loc["2019-04-01":"2019-04-26"].copy(),
                    windpower.loc["2019-05-01":"2019-05-27"].copy(),
                    windpower.loc["2019-06-01":"2019-06-26"].copy(),
                ]
            )

            self.windpower_valid = pd.concat(
                [
                    windpower.loc["2019-01-28":"2019-01-31"].copy(),
                    windpower.loc["2019-02-25":"2019-02-28"].copy(),
                    windpower.loc["2019-03-28":"2019-03-31"].copy(),
                    windpower.loc["2019-04-27":"2019-04-30"].copy(),
                    windpower.loc["2019-05-28":"2019-05-31"].copy(),
                    windpower.loc["2019-06-27":"2019-06-30"].copy(),
                ]
            )

            self.windpower_test = windpower.loc["2019-07-01":"2019-09-30"].copy()

        elif self.experiment == 2:
            self.windpower_train = pd.concat(
                [
                    windpower.loc["2019-04-01":"2019-04-26"].copy(),
                    windpower.loc["2019-05-01":"2019-05-27"].copy(),
                    windpower.loc["2019-06-01":"2019-06-26"].copy(),
                    windpower.loc["2019-07-01":"2019-07-27"].copy(),
                    windpower.loc["2019-08-01":"2019-08-27"].copy(),
                    windpower.loc["2019-09-01":"2019-09-26"].copy(),
                ]
            )

            self.windpower_valid = pd.concat(
                [
                    windpower.loc["2019-04-27":"2019-04-30"].copy(),
                    windpower.loc["2019-05-28":"2019-05-31"].copy(),
                    windpower.loc["2019-06-27":"2019-06-30"].copy(),
                    windpower.loc["2019-07-28":"2019-07-31"].copy(),
                    windpower.loc["2019-08-28":"2019-08-31"].copy(),
                    windpower.loc["2019-09-27":"2019-09-30"].copy(),
                ]
            )

            self.windpower_test = windpower.loc["2019-10-01":"2019-12-31"].copy()

        elif self.experiment == 3:
            self.windpower_train = pd.concat(
                [
                    windpower.loc["2019-07-01":"2019-07-27"].copy(),
                    windpower.loc["2019-08-01":"2019-08-27"].copy(),
                    windpower.loc["2019-09-01":"2019-09-26"].copy(),
                    windpower.loc["2019-10-01":"2019-10-27"].copy(),
                    windpower.loc["2019-11-01":"2019-11-26"].copy(),
                    windpower.loc["2019-12-01":"2019-12-27"].copy(),
                ]
            )

            self.windpower_valid = pd.concat(
                [
                    windpower.loc["2019-07-28":"2019-07-31"].copy(),
                    windpower.loc["2019-08-28":"2019-08-31"].copy(),
                    windpower.loc["2019-09-27":"2019-09-30"].copy(),
                    windpower.loc["2019-10-28":"2019-10-31"].copy(),
                    windpower.loc["2019-11-27":"2019-11-30"].copy(),
                    windpower.loc["2019-12-28":"2019-12-31"].copy(),
                ]
            )

            self.windpower_test = windpower.loc["2020-01-01":"2020-03-31"].copy()

        elif self.experiment == 4:
            self.windpower_train = pd.concat(
                [
                    windpower.loc["2019-10-01":"2019-10-27"].copy(),
                    windpower.loc["2019-11-01":"2019-11-26"].copy(),
                    windpower.loc["2019-12-01":"2019-12-27"].copy(),
                    windpower.loc["2020-01-01":"2020-01-27"].copy(),
                    windpower.loc["2020-02-01":"2020-02-25"].copy(),
                    windpower.loc["2020-03-01":"2020-03-27"].copy(),
                ]
            )

            self.windpower_valid = pd.concat(
                [
                    windpower.loc["2019-10-28":"2019-10-31"].copy(),
                    windpower.loc["2019-11-27":"2019-11-30"].copy(),
                    windpower.loc["2019-12-28":"2019-12-31"].copy(),
                    windpower.loc["2020-01-28":"2020-01-31"].copy(),
                    windpower.loc["2020-02-26":"2020-02-29"].copy(),
                    windpower.loc["2020-03-28":"2020-03-31"].copy(),
                ]
            )

            self.windpower_test = windpower.loc["2020-04-01":"2020-06-30"].copy()

        elif self.experiment == 5:
            self.windpower_train = pd.concat(
                [
                    windpower.loc["2020-01-01":"2020-01-27"].copy(),
                    windpower.loc["2020-02-01":"2020-02-25"].copy(),
                    windpower.loc["2020-03-01":"2020-03-27"].copy(),
                    windpower.loc["2020-04-01":"2020-04-26"].copy(),
                    windpower.loc["2020-05-01":"2020-05-27"].copy(),
                    windpower.loc["2020-06-01":"2020-06-26"].copy(),
                ]
            )

            self.windpower_valid = pd.concat(
                [
                    windpower.loc["2020-01-28":"2020-01-31"].copy(),
                    windpower.loc["2020-02-26":"2020-02-29"].copy(),
                    windpower.loc["2020-03-28":"2020-03-31"].copy(),
                    windpower.loc["2020-04-27":"2020-04-30"].copy(),
                    windpower.loc["2020-05-28":"2020-05-31"].copy(),
                    windpower.loc["2020-06-27":"2020-06-30"].copy(),
                ]
            )

            self.windpower_test = windpower.loc["2020-07-01":"2020-09-30"].copy()

        elif self.experiment == 6:
            self.windpower_train = pd.concat(
                [
                    windpower.loc["2020-04-01":"2020-04-26"].copy(),
                    windpower.loc["2020-05-01":"2020-05-27"].copy(),
                    windpower.loc["2020-06-01":"2020-06-26"].copy(),
                    windpower.loc["2020-07-01":"2020-07-27"].copy(),
                    windpower.loc["2020-08-01":"2020-08-27"].copy(),
                    windpower.loc["2020-09-01":"2020-09-26"].copy(),
                ]
            )

            self.windpower_valid = pd.concat(
                [
                    windpower.loc["2020-04-27":"2020-04-30"].copy(),
                    windpower.loc["2020-05-28":"2020-05-31"].copy(),
                    windpower.loc["2020-06-27":"2020-06-30"].copy(),
                    windpower.loc["2020-07-28":"2020-07-31"].copy(),
                    windpower.loc["2020-08-28":"2020-08-31"].copy(),
                    windpower.loc["2020-09-27":"2020-09-30"].copy(),
                ]
            )

            self.windpower_test = windpower.loc["2020-10-01":"2020-12-31"].copy()

        elif self.experiment == 7:
            self.windpower_train = pd.concat(
                [
                    windpower.loc["2020-07-01":"2020-07-27"].copy(),
                    windpower.loc["2020-08-01":"2020-08-27"].copy(),
                    windpower.loc["2020-09-01":"2020-09-26"].copy(),
                    windpower.loc["2020-10-01":"2020-10-27"].copy(),
                    windpower.loc["2020-11-01":"2020-11-26"].copy(),
                    windpower.loc["2020-12-01":"2020-12-27"].copy(),
                ]
            )

            self.windpower_valid = pd.concat(
                [
                    windpower.loc["2020-07-28":"2020-07-31"].copy(),
                    windpower.loc["2020-08-28":"2020-08-31"].copy(),
                    windpower.loc["2020-09-27":"2020-09-30"].copy(),
                    windpower.loc["2020-10-28":"2020-10-31"].copy(),
                    windpower.loc["2020-11-27":"2020-11-30"].copy(),
                    windpower.loc["2020-12-28":"2020-12-31"].copy(),
                ]
            )

            self.windpower_test = windpower.loc["2021-01-01":"2021-03-31"].copy()

        elif self.experiment == 8:
            self.windpower_train = pd.concat(
                [
                    windpower.loc["2020-10-01":"2020-10-27"].copy(),
                    windpower.loc["2020-11-01":"2020-11-26"].copy(),
                    windpower.loc["2020-12-01":"2020-12-27"].copy(),
                    windpower.loc["2021-01-01":"2021-01-27"].copy(),
                    windpower.loc["2021-02-01":"2021-02-24"].copy(),
                    windpower.loc["2021-03-01":"2021-03-27"].copy(),
                ]
            )

            self.windpower_valid = pd.concat(
                [
                    windpower.loc["2020-10-28":"2020-10-31"].copy(),
                    windpower.loc["2020-11-27":"2020-11-30"].copy(),
                    windpower.loc["2020-12-28":"2020-12-31"].copy(),
                    windpower.loc["2021-01-28":"2021-01-31"].copy(),
                    windpower.loc["2021-02-25":"2021-02-28"].copy(),
                    windpower.loc["2021-03-28":"2021-03-31"].copy(),
                ]
            )

            self.windpower_test = windpower.loc["2021-04-01":"2021-06-30"].copy()

        ### wind speed data
        windspeed_train = windspeed.loc[
            windspeed.index.get_level_values("prediction_time [UTC]").isin(
                self.windpower_train.index
            )
        ].copy()
        windspeed_valid = windspeed.loc[
            windspeed.index.get_level_values("prediction_time [UTC]").isin(
                self.windpower_valid.index
            )
        ].copy()
        windspeed_test = windspeed.loc[
            windspeed.index.get_level_values("prediction_time [UTC]").isin(
                self.windpower_test.index
            )
        ].copy()

        ## Standardize wind speed data
        ### Calculate mean and standard deviation based on training data (for the entire dataframe)
        self.mean_windspeed = windspeed_train.stack().mean()
        self.std_windspeed = windspeed_train.stack().std()

        self.windspeed_train = standardize(
            windspeed_train, self.mean_windspeed, self.std_windspeed
        )
        self.windspeed_valid = standardize(
            windspeed_valid, self.mean_windspeed, self.std_windspeed
        )
        self.windspeed_test = standardize(
            windspeed_test, self.mean_windspeed, self.std_windspeed
        )

        # Create datasets for training, validation and testing
        self.train_dataset = WPF_Germany_Dataset(
            windspeed=self.windspeed_train,
            windpower=self.windpower_train,
            forecast_horizon=self.forecast_horizon,
            n_past_timesteps=self.n_past_timesteps,
        )
        self.val_dataset = WPF_Germany_Dataset(
            windspeed=self.windspeed_valid,
            windpower=self.windpower_valid,
            forecast_horizon=self.forecast_horizon,
            n_past_timesteps=self.n_past_timesteps,
        )
        self.test_dataset = WPF_Germany_Dataset(
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
