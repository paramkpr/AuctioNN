"""
campaign.py
This module provides a Campaign class that represents the campaigns that the
media company is running. It is also responsible for bootstrapping the campaigns
for simulation purposes.
"""

from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np


@dataclass(slots=True)
class Campaign:
    """
    Represents a campaign that the media company is running.
    """

    id: str
    value_per_conv: float
    target_cpa: float | None
    budget_remaining: float
    ad_stock: defaultdict[int, int]  # user_id → exposure count
    true_conv_rate: float | None


def bootstrap_campaigns(
    clean_data_path: Path = Path("./data/cleaned/test/"),
    seed: int = 42,
) -> list[Campaign]:
    """
    First, from clean_data we load the list of campaign ids that we have.
    Then, for each campaign, we create a Campaign object with synthetic,
    random values for the campaign parameters. We provide a seed for reproducibility.
    """
    # Load the campaign ids from the clean data
    campaign_ids = [
        8334,
        13411,
        13505,
        14108,
        14213,
        14546,
        16007,
        17562,
        18997,
        19441,
        19442,
        40582,
        41142,
        42252,
        42300,
        42388,
        42485,
        42488,
        42517,
        42540,
        42569,
        42580,
        42593,
        42751,
        42838,
        42844,
        42915,
        42943,
        42993,
        43013,
        43015,
        43102,
        43247,
        43249,
        43423,
        43633,
        43662,
        43787,
        43789,
        43813,
        44002,
        44120,
        44126,
        44165,
        44424,
        44584,
        44729,
        44736,
        44806,
        44867,
        44923,
        45363,
        45432,
        45457,
        45459,
        45460,
        45461,
        45482,
        45488,
        45783,
        46536,
        46729,
        46975,
        47009,
        47068,
        47086,
        47118,
        47120,
        47170,
        47191,
        47193,
        47205,
        47242,
        47245,
        47253,
        47259,
        47322,
        47362,
        47381,
        47386,
        47451,
        47455,
        47462,
        47465,
        47548,
        47586,
        47589,
        47663,
    ]

    rng = np.random.default_rng(seed)

    true_conv_rates = {
        "47465": 0.3102387576868887,
        "47118": 0.266126886177399,
        "8334": 0.12859463521804643,
        "43813": 0.07361177813387486,
        "47120": 0.04503834935070243,
        "14546": 0.032115538664350185,
        "42540": 0.030070537970513234,
        "45459": 0.028994399063502912,
        "47455": 0.02703381750325254,
        "43249": 0.026218913641306806,
        "45482": 0.023923961258857426,
        "44584": 0.02224011016382986,
        "43787": 0.01978860081013625,
        "42252": 0.019003305856087928,
        "47068": 0.018538771554439067,
        "47381": 0.018020652280831863,
        "42580": 0.0141764642346549,
        "45460": 0.010825483813828841,
        "45461": 0.010250525068647773,
        "42943": 0.010176279011318979,
        "42838": 0.009837966707980397,
        "44120": 0.00952769187271262,
        "43015": 0.008268285068707642,
        "42915": 0.007852748509365553,
        "13411": 0.007534713209500203,
        "47386": 0.007329249883097833,
        "47589": 0.006761700990471277,
        "44126": 0.006121870596852269,
        "42300": 0.005827087421672,
        "43102": 0.005756558527676142,
        "14108": 0.004646126682339851,
        "19442": 0.0045685317409677444,
        "42488": 0.004468240504607357,
        "45432": 0.0041931156856325055,
        "47462": 0.004185750221588819,
        "42993": 0.004119440014825516,
        "45363": 0.004008599651786829,
        "47451": 0.0037730761880186894,
        "40582": 0.003703130709625313,
        "43662": 0.0036267771645030443,
        "41142": 0.0034644916607822515,
        "46975": 0.003412104139103758,
        "42517": 0.0031534156703856443,
        "14213": 0.003100967630226346,
        "43423": 0.003098831429864298,
        "43247": 0.0026983031063624453,
        "42751": 0.0026074077471001035,
        "42844": 0.0023530929854233586,
        "47253": 0.0023221113917591017,
        "45457": 0.001991442636191973,
        "47362": 0.0019510221438571202,
        "44002": 0.00188449221952546,
        "44424": 0.0018136889725151822,
        "44165": 0.0017450830100427547,
        "44736": 0.0016600595140415072,
        "16007": 0.001635877946302602,
        "19441": 0.0015653505448282423,
        "18997": 0.0015629586414671463,
        "42388": 0.0014693976219667666,
        "47586": 0.001366134672048329,
        "45488": 0.0013345396008273976,
        "42485": 0.0009819858974037517,
        "43013": 0.0009803364689223704,
        "47242": 0.0009722531439631271,
        "46729": 0.000778631927821764,
        "44867": 0.0007609033089721649,
        "47322": 0.0007246703239668061,
        "47191": 0.0006766825762632178,
        "47259": 0.0006357040238295083,
        "44729": 0.0006325953898054719,
        "46536": 0.0005950709836518195,
        "47170": 0.0005705501522452478,
        "47193": 0.0004532562135115081,
        "44923": 0.00038934420115818713,
        "17562": 0.00038526137382663254,
        "47086": 0.00034393338040858065,
        "43633": 0.00027114348652487513,
        "47663": 0.00023117412746585735,
        "42593": 0.00019428664054385277,
        "47245": 0.00015451046719955087,
        "44806": 0.00011666282810033447,
        "47205": 0.00011322897686296507,
        "42569": 0.00010047086908092346,
        "45783": 9.326308689169305e-05,
        "47548": 7.217373150672176e-05,
        "47009": 5.037789315148986e-05,
        "43789": 1.08300559365865e-05,
        "13505": 1.365209183904162e-08,
    }

    # Create a list of Campaign objects
    campaigns = []
    for campaign_id in campaign_ids:
        true_conv_rate = true_conv_rates[str(campaign_id)]
        value_per_conv = float(
            np.clip(rng.uniform(0.8, 1.2) / true_conv_rate, 0.50, 500.0)
        )
        target_cpa = rng.uniform(
            1.1 * value_per_conv, 3.0 * value_per_conv
        )  # ignore for now
        # budget_remaining = rng.uniform(500.0, 5000.0)  # total $ budget
        budget_remaining = 1000.0  # total $ budget
        ad_stock = defaultdict(int)

        campaigns.append(
            Campaign(
                id=campaign_id,
                true_conv_rate=true_conv_rate,
                value_per_conv=value_per_conv,
                target_cpa=target_cpa,
                budget_remaining=budget_remaining,
                ad_stock=ad_stock,
            )
        )

    return campaigns
