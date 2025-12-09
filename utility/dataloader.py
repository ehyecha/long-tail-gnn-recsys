from collections import Counter

def get_item_popularity(train):
        counter = Counter()
        for item in train:
            counter[item] += 1
        return counter

def split_head_tail_items_by_cumulative_popularity(item_counts, volume_threshold=0.8):
        sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
        total_interactions = sum(count for _, count in sorted_items)

        cumulative = 0
        head_items = set()
        for item, count in sorted_items:
            cumulative += count
            head_items.add(item)
            if cumulative / total_interactions >= volume_threshold:
                break
        tail_items = set(item_counts.keys()) - head_items
        return head_items, tail_items