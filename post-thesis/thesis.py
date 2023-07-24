import arxiv


class Arxiv:
    max_results = 50

    @classmethod
    def get_list(self, keyword):
        """
        arXiv APIから論文情報を取得する
        """

        # 検索条件
        query = "ti:%22" + keyword + "%22"

        result = arxiv.Search(
            query=query,
            max_results=self.max_results,
            sort_order=arxiv.SortOrder.Descending,  # 投稿日の降順
            sort_by=arxiv.SortCriterion.SubmittedDate,  # 投稿日でソート
        )

        return self.categorize(result.results())

    @staticmethod
    def categorize(result):
        """
        第1カテゴリがComputer Science, Mathematics, Statisticsのみを抽出
        カテゴリ一覧: https://arxiv.org/category_taxonomy
        """
        ret = []
        for r in list(result):
            if "cs" in r.primary_category or "math" in r.primary_category or "stat" in r.primary_category:
                ret.append(r)
        return ret
