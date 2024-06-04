from data import DATA_DIR
import os
import pandas as pd
from haversine import haversine

def from_str_to_tuple(lat_lng):
    return tuple([float(x) for x in lat_lng.split(",")])


class DataHandler:
    def __init__(
        self,
        data_path=os.path.join(DATA_DIR, "user_click_dataset.csv"),
        targets=["booked_flight", "booked_hotel", "booked_rental"],
    ):
        self.data_path = data_path
        self.targets = targets
        self.data = None
        self.min_distance_without_flight = None
        self.min_distance_without_flight = None

    def load_data(self):
        self.data = pd.read_csv(self.data_path)

    def compute_distance(self):
        self.data["distance"] = self.data.apply(
            lambda row: haversine(
                from_str_to_tuple(row["user_lat_lng"]),
                from_str_to_tuple(row["dest_lat_lng"]),
            ),
            axis=1,
        )

    def set_min_max_distance_without_flight(self):
        data_extract1 = self.data[
            (self.data["booked_flight"] == 0)
            & ((self.data["booked_hotel"] == 1) | (self.data["booked_rental"] == 1))
        ]
        data_extract2 = self.data[self.data["booked_flight"] == 1]
        
        self.min_distance_without_flight = data_extract1["distance"].min()
        self.min_distance_with_flight = data_extract2["distance"].min()

    def fill_missing_flights(self):
        condition = self.data["distance"] < self.min_distance_without_flight
        self.data.loc[condition, "booked_flight"] = self.data.loc[condition, "booked_flight"].fillna(0)

    def fill_missing_bookings(self):
        condition = (self.data["distance"] > self.min_distance_with_flight) & (
            self.data["booked_flight"] == 0
        )
        self.data.loc[condition, ["booked_hotel", "booked_rental"]] = self.data.loc[condition,
            ["booked_hotel", "booked_rental"]].fillna(0)

        self.data["booked_hotel"] = self.data["booked_hotel"].fillna(
            1 - self.data["booked_rental"]
        )
        self.data["booked_rental"] = self.data["booked_rental"].fillna(
            1 - self.data["booked_hotel"]
        )

    def fffill(self):
        columns = [c for c in self.data.columns if c not in self.targets]
        self.data[columns] = self.data[columns].ffill()

    def flatten_lat_lng(self):
        self.data["user_lat"] = self.data["user_lat_lng"].apply(lambda x: x.split(",")[0])
        self.data["user_lng"] = self.data["user_lat_lng"].apply(lambda x: x.split(",")[1])
        self.data["dest_lat"] = self.data["dest_lat_lng"].apply(lambda x: x.split(",")[0])
        self.data["dest_lng"] = self.data["dest_lat_lng"].apply(lambda x: x.split(",")[1])
        self.data.drop(["user_lat_lng", "dest_lat_lng"], axis=1, inplace=True)

    def load_and_validate_data(self):
        self.load_data()
        self.fffill()
        self.compute_distance()
        self.set_min_max_distance_without_flight()
        self.fill_missing_flights()
        self.fill_missing_bookings()
        self.flatten_lat_lng()
        self.data.dropna(inplace=True)
