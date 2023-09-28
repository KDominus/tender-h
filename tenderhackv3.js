const axios = require('axios');
const CORE_ID = '';
const {
  Telegraf,
  Scenes,
  Context,
  Extra,
  Markup,
  Stage,
  state,
  session,
} = require('telegraf');
const { CallbackData } = require('@bot-base/callback-data');

const bot = new Telegraf(CORE_ID);

bot.use(session());

////files assets ///

const file1 =
  './assets/files/Инструкция по работе с банковскими гарантиями.pdf';
const file2 =
  './assets/files/Инструкция по работе с машиночитаемыми доверенностями.pdf';
const file3 =
  './assets/files/Инструкция по работе с Порталом для поставщика.pdf';
const file4 = './assets/files/Инструкция по работе с сервисом Логистика.pdf';
const file5 = './assets/files/Инструкция по созданию оферты и СТЕ.pdf';
const file6 = './assets/files/Инструкция по формированию YML.pdf';
const file7 = './assets/files/Инструкция по электронному актированию.pdf';
const file8 =
  './assets/files/Описание сервиса загрузки исполнении из системы учета.docx';
const file9 = './assets/files/Регламент.pdf';
const file10 =
  './assets/files/Руководство пользователя по импорту данных на УПД.docx';

//file query matcher
function fileQueryMatcher(recivedString) {
  switch (recivedString) {
    case 'Инструкция по работе с банковскими гарантиями':
      return file1;
      break;

    case 'Инструкция по работе с машиночитаемыми доверенностями':
      return file2;
      break;
    case 'Инструкция по работе с Порталом для поставщика':
      return file3;
      break;
    case 'Инструкция по работе с сервисом Логистика':
      return file4;
      break;
    case 'Инструкция по созданию оферты и СТЕ':
      return file5;
      break;
    case 'Инструкция по формированию YML':
      return file6;
      break;
    case 'Инструкция по электронному актированию':
      return file7;
      break;
    case 'Описание сервиса загрузки исполнении из системы учета':
      return file8;
      break;
    case 'Регламент':
      return file9;
      break;
    case 'Руководство пользователя по импорту данных на УПД':
      return file10;
      break;

    default:
      return false;
      break;
  }
}

////Icons sections////
const smileEm = '😃';
const ballonEm = '🎈';
const presentEm = '🎁';
const orangeEm = '🍊';
const snowmanEm = '⛄';
const crystallBallEm = '🔮';
const bunnyEm = '🐇';
const diamondEm = '💎';
const heartEm = '💓';
const performingArtsEm = '🎭';
const confetiBallEm = '🎊';
const partyPopperEm = '🎉';
const santaEm = '🎅';
const cakeEm = '🎂';
const knotBowEm = '🎀';
const vortexEm = '🌀';
const inLoveEm = '😍';
const cristmasTreeEm = '🎄';

const buttonAskQuestion = `${vortexEm} Поиск источника`;
const buttonSearchSource = `${smileEm}  Задать вопрос`;

const aviableDatesArray = [buttonAskQuestion, buttonSearchSource];

//////keyboards/////

const actionKbd = {
  reply_markup: JSON.stringify({
    keyboard: [
      [
        { text: `${buttonSearchSource}`, callback_data: 'datesKbdCB' },
        { text: `${buttonAskQuestion}`, callback_data: 'datesKbdCB' },
      ],
    ],
    resize_keyboard: true,
  }),
};

const toMainMenuKbd = {
  reply_markup: JSON.stringify({
    keyboard: [[{ text: `${cakeEm} На главную`, callback_data: '' }]],
    resize_keyboard: true,
  }),
};

//Scenes

////////////main screen scene////////////////
const mainScreen = new Scenes.BaseScene('MAIN_SCREEN_SCENE');
mainScreen.enter(async (ctx) => {
  await ctx.reply('Выберите опцию', actionKbd);

  await mainScreen.on('message', async (ctx) => {
    if (
      ctx.message.text !== buttonAskQuestion &&
      ctx.message.text !== buttonSearchSource
    ) {
      ctx.reply('выберите одну из доступных кнопок');
      ctx.scene.reenter();
    } else if (ctx.message.text === buttonAskQuestion) {
      ctx.scene.enter('ASK_QUESTION_SCENE');
    } else if (ctx.message.text === buttonSearchSource) {
      ctx.scene.enter('SEARCH_SOURCE_SCENE');
    } else {
      ctx.scene.reenter();
    }
  });
});

//custom scenes

/////////ASK QUESTION SCENE
const askQuestionScene = new Scenes.BaseScene('ASK_QUESTION_SCENE');
askQuestionScene.enter(async (ctx) => {
  await ctx.reply(`Задайте вопрос:`, toMainMenuKbd);

  await askQuestionScene.on('text', async (ctx) => {
    if (ctx.message.text === `${cakeEm} На главную`) {
      await ctx.scene.enter('MAIN_SCREEN_SCENE');
    } else {
      /////////// REQUEST//////////////
      const data = await axios
        .post(
          'http://127.0.0.1:8000/get_answer_pdf',
          {
            text: ctx.message.text,
          },
          {
            headers: {
              'Content-Type': 'application/json',
            },
          },
        )
        .then(async (data) => {
          const fileMacherString = data.data.answer.split('\n')[0];
          const isFileRecordExists = fileQueryMatcher(fileMacherString);
          await ctx.reply(data.data.answer);
          ctx.reply('Ожидайте скачивания файла');
          if (isFileRecordExists !== false) {
            ctx.replyWithDocument({ source: isFileRecordExists });
          }
        })
        .catch((error) => console.log(error));

      await ctx.scene.reenter();
    }
  });
});

///////SEARCH_SOURCE_SCENE
const searchSourceScene = new Scenes.BaseScene('SEARCH_SOURCE_SCENE');
searchSourceScene.enter(async (ctx) => {
  await ctx.reply(`Задайте строку для поиска:`, toMainMenuKbd);

  await searchSourceScene.on('text', async (ctx) => {
    if (ctx.message.text === `${cakeEm} На главную`) {
      await ctx.scene.enter('MAIN_SCREEN_SCENE');
    } else {
      /////////// REQUEST//////////////
      const data = await axios
        .post(
          'http://127.0.0.1:8000/get_answer_xlsx',
          {
            text: ctx.message.text,
          },
          {
            headers: {
              'Content-Type': 'application/json',
            },
          },
        )
        .then((data) => {
          ctx.reply(data.data.answer);
        })
        .catch((error) => console.log(error));

      await ctx.scene.reenter();
    }
  });
});

//////Stages Sections//////
const stage = new Scenes.Stage([
  mainScreen,
  askQuestionScene,
  searchSourceScene,
]);

bot.use(stage.middleware());

////utility section////
bot.start(async (ctx) => {
  await ctx.scene.enter('MAIN_SCREEN_SCENE');
});

bot.on('message', async (ctx) => {
  await ctx.scene.enter('MAIN_SCREEN_SCENE');
});

bot
  .launch()
  .then(() => console.log('-=Active=-'))
  .catch((err) => console.log(err));

process.once('SIGINT', () => bot.stop('SIGINT'));
process.once('SIGTERM', () => bot.stop('SIGTERM'));
