import itertools
import numpy as np
from typing import List

from collections import defaultdict


class APriori:
    def __init__(self, transactions: List[List], min_support, min_confidence, min_leverage,
                 print_slice_size=200):
        self.transactions = transactions
        self.transactions_number = len(self.transactions)
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_leverage = min_leverage
        self.frequent_sets = None
        self.print_slice_size = print_slice_size
        self.support = defaultdict(float)
        self.rules = {}

    def _filter_support(self, candidate_set):
        set_count = defaultdict(int)
        processed_items = 0
        for items in candidate_set:
            for transaction in self.transactions:
                if items.issubset(transaction):
                    set_count[items] += 1
            processed_items += 1
            if processed_items % self.print_slice_size == 0:
                print(f"\tProcessed {processed_items} of {len(candidate_set)} items")
        frequent_items = {items for items in candidate_set
                          if set_count[items] / self.transactions_number >= self.min_support}
        self.support.update({items: set_count[items] / self.transactions_number
                             for items in frequent_items})
        return frequent_items

    def _find_single_element_frequent_sets(self):
        all_products = {frozenset({item}) for item in itertools.chain.from_iterable(self.transactions)}

        frequent_items = self._filter_support(all_products)
        return frequent_items

        # flat_transactions = list(itertools.chain.from_iterable(self.transactions))
        # print(f"number of candidates of length 1: {self.transactions_number}")
        # products_counts = np.bincount(flat_transactions)
        # print("products counted")
        # frequent_items = np.where(products_counts / self.transactions_number >= self.min_support)[0]
        # print("frequent_items created")
        # existing_items = np.where(products_counts > 0)[0]
        # print("existing_items created")
        # support_values = products_counts[existing_items] / self.transactions_number
        # print("support_values created")
        # support_keys = [frozenset({i}) for i in flat_transactions]
        # print("support_keys created")
        # support_dict = dict(zip(support_keys, support_values))
        # print("supports_dict created")
        # self.support.update(support_dict)
        # print("support updated")
        # return {frozenset({item}) for item in frequent_items}

    def _generate_next_candidate_sets(self, frequent_sets):
        current_sets_length = len(frequent_sets) + 1
        return set([x.union(y) for x in frequent_sets[-1] for y in frequent_sets[-1]
                    if len(x.union(y)) == current_sets_length])

    def find_frequent_sets(self):
        frequent_sets = [self._find_single_element_frequent_sets()]

        while frequent_sets[-1] != set():
            canditade_sets = self._generate_next_candidate_sets(frequent_sets)
            print(f"Number of candidates of length {len(frequent_sets) + 1}: {len(canditade_sets)}")
            frequent_sets.append(self._filter_support(canditade_sets))

        self.frequent_sets = set.union(*frequent_sets)
        return self.frequent_sets

    def _create_rule(self, product_set, full_product_set):
        for product in product_set:
            remaining_products = product_set - {product}
            if remaining_products:
                current_product_set = full_product_set - remaining_products
                support = self.support[frozenset.union(current_product_set, remaining_products)]
                confidence = support / self.support[remaining_products]
                leverage = (support -
                            self.support[remaining_products] * self.support[current_product_set])
                if (confidence >= self.min_confidence and leverage >= self.min_leverage and
                        (remaining_products, current_product_set) not in self.rules):
                    self.rules[(remaining_products, current_product_set)] = (
                        confidence, leverage, support)

                    self._create_rule(remaining_products, full_product_set)

    def find_association_rules(self, frequent_sets=None, min_confidence=None, min_leverage=None):
        if frequent_sets is None:
            frequent_sets = self.frequent_sets
        if min_confidence is not None:
            self.min_confidence = min_confidence
        if min_leverage is not None:
            self.min_leverage = min_leverage

        for products in frequent_sets:
            self._create_rule(products, products)


if __name__ == "__main__":
    data = [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
    a_priori = APriori(data, min_support=0.5, min_confidence=0.7, min_leverage=0.01)
    print(a_priori.find_frequent_sets())
    a_priori.find_association_rules()
    for k, v in a_priori.rules.items():
        print(f"{tuple(k[0])} => {tuple(k[1])}\n{v}\n\n")

    dataset = [
        ['bread', 'milk'],
        ['bread', 'diaper', 'beer', 'egg'],
        ['milk', 'diaper', 'beer', 'cola'],
        ['bread', 'milk', 'diaper', 'beer'],
        ['bread', 'milk', 'diaper', 'cola'],
    ]
    a_priori2 = APriori(dataset, min_support=0.6, min_confidence=0.6, min_leverage=0.01)
    print(a_priori2.find_frequent_sets())
    a_priori2.find_association_rules()
    for k, v in a_priori2.rules.items():
        print(f"{tuple(k[0])} => {tuple(k[1])}\n{v}\n\n")