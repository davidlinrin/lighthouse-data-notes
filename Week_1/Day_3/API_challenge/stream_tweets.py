{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "import twitter\n",
    "import json\n",
    "import os\n",
    "import os.path"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Task:** Load the values of access tokens and keys from environmental variables to python variables"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "kKfjs36cllF9UGKWZRNVn4yuQ\n",
      "SmjSYB0b72PpiwETxh1xrHwowbUJBjWsq4NFqoPhsHWMNNxNyT\n",
      "1286030070742564864-ALj25LxvrDJa8KtkUind8UjRsSvnnn\n",
      "u9qLD9FRrDO0nEMRVPwx1MWKmxpOVpEPmlZzSGGMhtj1O\n"
     ]
    }
   ],
   "source": [
    "consumer_key = os.environ['consumer_key']\n",
    "consumer_secret= os.environ['consumer_secret']\n",
    "access_token = os.environ['access_token_key']\n",
    "access_token_secret = os.environ['access_token_secret']\n",
    "\n",
    "print(consumer_key)\n",
    "print(consumer_secret)\n",
    "print(access_token)\n",
    "print(access_token_secret)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "api = twitter.Api(consumer_key=consumer_key,\n",
    "                  consumer_secret=consumer_secret,\n",
    "                  access_token_key=access_token,\n",
    "                  access_token_secret=access_token_secret)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'twitter.api.Api'>\n"
     ]
    }
   ],
   "source": [
    "# Checking the type of api object\n",
    "print(type(api))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "metadata": {},
   "outputs": [],
   "source": [
    "## FOLLOWING FUNCTION WILL COLLECT REAL-TIME TWEETS IN OUR COMPUTER\n",
    "\n",
    "# data returned will be for any tweet mentioning strings in the list FILTER\n",
    "FILTER = ['#datascience']\n",
    "\n",
    "# Languages to filter tweets by is a list. This will be joined by Twitter\n",
    "# to return data mentioning tweets only in the english language.\n",
    "LANGUAGES = ['en']\n",
    "\n",
    "location = '/mnt/d/lighthouse/lighthouse_data_notes/Week_1/Day_3/'\n",
    "\n",
    "def main(location, FILTER, LANGUAGES):\n",
    "    name = 'test.txt'\n",
    "    file_name = location + name\n",
    "    with open(file_name, 'a') as f:\n",
    "        # api.GetStreamFilter will return a generator that yields one status\n",
    "        # message (i.e., Tweet) at a time as a JSON dictionary.\n",
    "        for line in api.GetStreamFilter(track=FILTER, languages=LANGUAGES, ):\n",
    "            f.write(json.dumps(line))\n",
    "            f.write('\\n')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "metadata": {
    "jupyter": {
     "outputs_hidden": true
    }
   },
   "outputs": [
    {
     "ename": "KeyboardInterrupt",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-92-77d9e3f7587d>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;31m# Execute function\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 2\u001b[0;31m \u001b[0mmain\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mlocation\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mFILTER\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mLANGUAGES\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;32m<ipython-input-90-85f50b7cfea3>\u001b[0m in \u001b[0;36mmain\u001b[0;34m(location, FILTER, LANGUAGES)\u001b[0m\n\u001b[1;32m     16\u001b[0m         \u001b[0;31m# api.GetStreamFilter will return a generator that yields one status\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     17\u001b[0m         \u001b[0;31m# message (i.e., Tweet) at a time as a JSON dictionary.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 18\u001b[0;31m         \u001b[0;32mfor\u001b[0m \u001b[0mline\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mapi\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mGetStreamFilter\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtrack\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mFILTER\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlanguages\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mLANGUAGES\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     19\u001b[0m             \u001b[0mf\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mwrite\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mjson\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdumps\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mline\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     20\u001b[0m             \u001b[0mf\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mwrite\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'\\n'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/.local/lib/python3.8/site-packages/twitter/api.py\u001b[0m in \u001b[0;36mGetStreamFilter\u001b[0;34m(self, follow, track, locations, languages, delimited, stall_warnings, filter_level)\u001b[0m\n\u001b[1;32m   4591\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   4592\u001b[0m         \u001b[0mresp\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_RequestStream\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0murl\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m'POST'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdata\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mdata\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 4593\u001b[0;31m         \u001b[0;32mfor\u001b[0m \u001b[0mline\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mresp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0miter_lines\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   4594\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0mline\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   4595\u001b[0m                 \u001b[0mdata\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_ParseAndCheckTwitter\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mline\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdecode\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'utf-8'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/lib/python3/dist-packages/requests/models.py\u001b[0m in \u001b[0;36miter_lines\u001b[0;34m(self, chunk_size, decode_unicode, delimiter)\u001b[0m\n\u001b[1;32m    792\u001b[0m         \u001b[0mpending\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    793\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 794\u001b[0;31m         \u001b[0;32mfor\u001b[0m \u001b[0mchunk\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0miter_content\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mchunk_size\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mchunk_size\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdecode_unicode\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mdecode_unicode\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    795\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    796\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0mpending\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/lib/python3/dist-packages/requests/models.py\u001b[0m in \u001b[0;36mgenerate\u001b[0;34m()\u001b[0m\n\u001b[1;32m    748\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0mhasattr\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mraw\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m'stream'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    749\u001b[0m                 \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 750\u001b[0;31m                     \u001b[0;32mfor\u001b[0m \u001b[0mchunk\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mraw\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mstream\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mchunk_size\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdecode_content\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mTrue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    751\u001b[0m                         \u001b[0;32myield\u001b[0m \u001b[0mchunk\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    752\u001b[0m                 \u001b[0;32mexcept\u001b[0m \u001b[0mProtocolError\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0me\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/lib/python3/dist-packages/urllib3/response.py\u001b[0m in \u001b[0;36mstream\u001b[0;34m(self, amt, decode_content)\u001b[0m\n\u001b[1;32m    558\u001b[0m         \"\"\"\n\u001b[1;32m    559\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mchunked\u001b[0m \u001b[0;32mand\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msupports_chunked_reads\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 560\u001b[0;31m             \u001b[0;32mfor\u001b[0m \u001b[0mline\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mread_chunked\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mamt\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdecode_content\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mdecode_content\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    561\u001b[0m                 \u001b[0;32myield\u001b[0m \u001b[0mline\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    562\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/lib/python3/dist-packages/urllib3/response.py\u001b[0m in \u001b[0;36mread_chunked\u001b[0;34m(self, amt, decode_content)\u001b[0m\n\u001b[1;32m    750\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    751\u001b[0m             \u001b[0;32mwhile\u001b[0m \u001b[0;32mTrue\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 752\u001b[0;31m                 \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_update_chunk_length\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    753\u001b[0m                 \u001b[0;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mchunk_left\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0;36m0\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    754\u001b[0m                     \u001b[0;32mbreak\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/lib/python3/dist-packages/urllib3/response.py\u001b[0m in \u001b[0;36m_update_chunk_length\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m    680\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mchunk_left\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    681\u001b[0m             \u001b[0;32mreturn\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 682\u001b[0;31m         \u001b[0mline\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_fp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mreadline\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    683\u001b[0m         \u001b[0mline\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mline\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msplit\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34mb\";\"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    684\u001b[0m         \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/lib/python3.8/socket.py\u001b[0m in \u001b[0;36mreadinto\u001b[0;34m(self, b)\u001b[0m\n\u001b[1;32m    667\u001b[0m         \u001b[0;32mwhile\u001b[0m \u001b[0;32mTrue\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    668\u001b[0m             \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 669\u001b[0;31m                 \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_sock\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mrecv_into\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mb\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    670\u001b[0m             \u001b[0;32mexcept\u001b[0m \u001b[0mtimeout\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    671\u001b[0m                 \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_timeout_occurred\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mTrue\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/lib/python3/dist-packages/urllib3/contrib/pyopenssl.py\u001b[0m in \u001b[0;36mrecv_into\u001b[0;34m(self, *args, **kwargs)\u001b[0m\n\u001b[1;32m    311\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mrecv_into\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    312\u001b[0m         \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 313\u001b[0;31m             \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mconnection\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mrecv_into\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    314\u001b[0m         \u001b[0;32mexcept\u001b[0m \u001b[0mOpenSSL\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mSSL\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mSysCallError\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0me\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    315\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msuppress_ragged_eofs\u001b[0m \u001b[0;32mand\u001b[0m \u001b[0me\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0margs\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0;34m(\u001b[0m\u001b[0;34m-\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m\"Unexpected EOF\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/lib/python3/dist-packages/OpenSSL/SSL.py\u001b[0m in \u001b[0;36mrecv_into\u001b[0;34m(self, buffer, nbytes, flags)\u001b[0m\n\u001b[1;32m   1819\u001b[0m             \u001b[0mresult\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0m_lib\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mSSL_peek\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_ssl\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mbuf\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnbytes\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1820\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1821\u001b[0;31m             \u001b[0mresult\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0m_lib\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mSSL_read\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_ssl\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mbuf\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnbytes\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1822\u001b[0m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_raise_ssl_error\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_ssl\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mresult\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1823\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m: "
     ]
    }
   ],
   "source": [
    "# Execute function\n",
    "main(location, FILTER, LANGUAGES)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Task:** Edit function `main` so it can store tweets anywhere (location specified as parameter). The FILTER and LANGUAGES should be parameters as well. Test it with different values and languages."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Task:** Create File `stream_tweets.py` that can be executed from the Terminal"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Task:** Start storing tweets with either happy smiley (`:)`) or sad smiley (`:(`). We will use this dataset during the NLP section."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [],
   "source": [
    "f = open(\"D:\\\\lighthouse\\lighthouse_data_notes\\Week_1\\Day_3\\test.txt\", \"w\")\n",
    "f.write(\"Hello my man\")\n",
    "f.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}