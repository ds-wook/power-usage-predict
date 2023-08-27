class Config:
    class data:
        path = "input/power-usage-predict/"
        encoder = "res/data/"
        target = "power_consumption"
        train = "train.csv"
        test = "test.csv"
        building_info = "building_info.csv"
        submit = "sample_submission.csv"
        n_splits = 5
        dataset_rename = {
            "건물번호": "building_number",
            "일시": "date_time",
            "기온(C)": "temperature",
            "강수량(mm)": "rainfall",
            "풍속(m/s)": "windspeed",
            "습도(%)": "humidity",
            "일조(hr)": "sunshine",
            "일사(MJ/m2)": "solar_radiation",
            "전력소비량(kWh)": "power_consumption",
        }
        building_info_rename = {
            "건물번호": "building_number",
            "건물유형": "building_type",
            "연면적": "total_area",
            "냉방면적(m2)": "cooling_area",
            "태양광용량(kW)": "solar_power_capacity",
            "ESS저장용량(kWh)": "ess_capacity",
            "POS용량(kW)": "pcs_capacity",
        }
        building_type_translation = {
            "건물기타": "Other Building",
            "공공": "Public",
            "대학교": "University",
            "데이터센터": "Data Center",
            "백화점및아울렛": "Department Store and Outlet",
            "병원": "Hospital",
            "상용": "Commercial",
            "아파트": "Apartment",
            "연구소": "Research Institute",
            "지식산업센터": "Knowledge Industry Center",
            "할인마트": "Discount Mart",
            "호텔및리조트": "Hotel and Resort",
        }
        categorical_features = ["building_number", "building_type"]

    class features:
        drop_train_features = ["sunshine", "solar_radiation"]
        drop_features = ["solar_power_capacity", "ess_capacity", "pcs_capacity", "sunshine", "solar_radiation"]
        categorical_features = ["building_type", "heat_index", "cluster"]
        numerical_features = ["temperature", "rainfall", "windspeed", "humidity", "total_area", "cooling_area"]
        lag_window = 3
