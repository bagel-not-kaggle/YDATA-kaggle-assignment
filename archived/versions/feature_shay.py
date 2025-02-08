def smooth_ctr(data, target_col,alpha=10):
    """Smooths the CTR by adding a prior."""
    clicks = data.groupby(target_col)['is_click'].sum().rename('clicks')
    views = data.groupby(target_col)['session_id'].count().rename('views')
    global_ctr = data['is_click'].mean()

    # Calculate smoothed CTR
    ctr = ((clicks + alpha * global_ctr) / (views + alpha)).rename('ctr')

    # Merge features
    data = pd.merge(data, ctr, on=target_col, how='left')
    data.fillna(global_ctr, inplace=True)  # Fill NaN with global CTR

    return data
      

def create_user_features(self, data, is_train):
        if is_train:
            # Calculate user-specific clicks and views
            user_clicks = data.groupby('user_id')['is_click'].sum().rename('user_clicks')
            self.user_views = data.groupby('user_id')['session_id'].count().rename('user_views')

            # Calculate smoothed user CTR
            self.user_ctr = ((user_clicks + self.alpha * self.global_ctr) / (self.user_views + self.alpha)).rename('user_ctr')

            # # Calculate user-campaign-specific clicks and views
            # user_campaign_clicks = data.groupby(['user_id', 'campaign_id'])['is_click'].sum().rename('campaign_clicks').reset_index()
            # self.user_campaign_views = data.groupby(['user_id', 'campaign_id'])['session_id'].count().rename('campaign_views').reset_index()

            # # Calculate smoothed user-campaign CTR
            # self.user_campaign_ctr = ((user_campaign_clicks['campaign_clicks'] + self.alpha * self.global_ctr) / (self.user_campaign_views['campaign_views'] + self.alpha)).rename('user_campaign_ctr')
            #
            # # Calculate user-product-specific clicks and views
            # user_product_clicks = data.groupby(['user_id', 'product'])['is_click'].sum().rename('product_clicks').reset_index()
            # self.user_product_views = data.groupby(['user_id', 'product'])['session_id'].count().rename('product_views').reset_index()
            #
            # # Calculate smoothed user-product CTR
            # self.user_product_ctr = ((user_product_clicks['product_clicks'] + self.alpha * self.global_ctr) / (self.user_product_views['product_views'] + self.alpha)).rename('user_product_ctr')
            #
            # # Calculate user-category-specific clicks and views
            # user_category_clicks = data.groupby(['user_id', 'product_category_1'])['is_click'].sum().rename('category_clicks').reset_index()
            # self.user_category_views = data.groupby(['user_id', 'product_category_1'])['session_id'].count().rename('category_views').reset_index()
            #
            # # Calculate smoothed user-category CTR
            # self.user_category_ctr = ((user_category_clicks['category_clicks'] + self.alpha * self.global_ctr) / (self.user_category_views['category_views'] + self.alpha)).rename('user_category_ctr')

        # Merge features
        data = pd.merge(data, self.user_ctr, on='user_id', how='left')
        data = pd.merge(data, self.user_views, on='user_id', how='left')
        # data = pd.merge(data, self.user_campaign_ctr, on=['user_id', 'campaign_id'], how='left')
        # data = pd.merge(data, self.user_campaign_views, on=['user_id', 'campaign_id'], how='left')
        # data = pd.merge(data, self.user_product_ctr, on=['user_id', 'product'], how='left')
        # data = pd.merge(data, self.user_product_views, on=['user_id', 'product'], how='left')
        # data = pd.merge(data, self.user_category_ctr, on=['user_id', 'product_category_1'], how='left')
        # data = pd.merge(data, self.user_category_views, on=['user_id', 'product_category_1'], how='left')

        data.fillna(0, inplace=True)  # Fill NaN (for users with no views) with 0

        return data

    def create_campaign_features(self, data, is_train):
        if is_train:
            # Calculate campaign-specific clicks and views
            campaign_clicks = data.groupby('campaign_id')['is_click'].sum().rename('campaign_clicks')
            self.campaign_views = data.groupby('campaign_id')['session_id'].count().rename('campaign_views')

            # Calculate smoothed campaign CTR
            self.campaign_ctr = ((campaign_clicks + self.alpha * self.global_ctr) / (self.campaign_views + self.alpha)).rename('campaign_ctr')

            # # Calculate campaign-product-specific clicks and views
            # campaign_product_clicks = data.groupby(['campaign_id', 'product'])['is_click'].sum().rename('product_clicks').reset_index()
            # self.campaign_product_views = data.groupby(['campaign_id', 'product'])['session_id'].count().rename('product_views').reset_index()
            #
            # # Calculate smoothed campaign-product CTR
            # self.campaign_product_ctr = ((campaign_product_clicks['product_clicks'] + self.alpha * self.global_ctr) / (self.campaign_product_views['product_views'] + self.alpha)).rename('campaign_product_ctr')

            # Merge features
        data = pd.merge(data, self.campaign_ctr, on='campaign_id', how='left')
        data = pd.merge(data, self.campaign_views, on='campaign_id', how='left')
        # data = pd.merge(data, self.campaign_product_ctr, on=['campaign_id', 'product'], how='left')
        # data = pd.merge(data, self.campaign_product_views, on=['campaign_id', 'product'], how='left')
        data.fillna(0, inplace=True)  # Fill NaN with 0

        return data

    def create_product_features(self, data, is_train):
        if is_train:
            # Calculate campaign-specific clicks and views
            campaign_clicks = data.groupby('campaign_id')['is_click'].sum().rename('campaign_clicks')
            self.campaign_views = data.groupby('campaign_id')['session_id'].count().rename('campaign_views')

            # Calculate smoothed campaign CTR
            self.campaign_ctr = ((campaign_clicks + self.alpha * self.global_ctr) / (self.campaign_views + self.alpha)).rename('campaign_ctr')

            # Calculate campaign-product-specific clicks and views
            # campaign_product_clicks = data.groupby(['campaign_id', 'product'])['is_click'].sum().rename('product_clicks').reset_index()
            # self.campaign_product_views = data.groupby(['campaign_id', 'product'])['session_id'].count().rename('product_views').reset_index()
            #
            # # Calculate smoothed campaign-product CTR
            # self.campaign_product_ctr = ((campaign_product_clicks['product_clicks'] + self.alpha * self.global_ctr) / (self.campaign_product_views['product_views'] + self.alpha)).rename('campaign_product_ctr')

            # Merge features
        data = pd.merge(data, self.campaign_ctr, on='campaign_id', how='left')
        data = pd.merge(data, self.campaign_views, on='campaign_id', how='left')
        # data = pd.merge(data, self.campaign_product_ctr, on=['campaign_id', 'product'], how='left')
        # data = pd.merge(data, self.campaign_product_views, on=['campaign_id', 'product'], how='left')

        data.fillna(0, inplace=True)  # Fill NaN with 0

        return data